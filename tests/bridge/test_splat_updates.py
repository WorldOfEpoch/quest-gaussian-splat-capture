import json
import pytest
import uuid
from unittest.mock import patch
from bridge.jobs import (Scheduler,set_splat_updates,splat_status,on_capture_received,
                         request_build,set_processing_paused,processing_paused,policy,_db,_depth_inventory)
from test_jobs import _seed,_capture

def test_live_splat_owns_capture_ingress_without_unpausing_full_queue(tmp_path):
    store,project,session=_seed(tmp_path)
    set_processing_paused(tmp_path,True)
    set_splat_updates(tmp_path,session['id'],True,native_orientation_reviewed=True)
    _capture(store,project,session,0)
    assert on_capture_received(tmp_path,session['id'])['status']=='live_splat'
    assert request_build(tmp_path,session['id'])['status']=='live_splat'
    assert processing_paused(tmp_path) is True
    assert policy(tmp_path,session['id'])['auto_enabled'] is False
    with _db(tmp_path) as db:
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0]==0
        first=db.execute('SELECT control_revision FROM splat_updates').fetchone()[0]
    set_splat_updates(tmp_path,session['id'],False)
    with _db(tmp_path) as db: assert db.execute('SELECT control_revision FROM splat_updates').fetchone()[0]!=first
    assert splat_status(tmp_path,session['id'])['status']=='paused'

def test_live_splat_waits_for_bootstrap_and_launches_once(tmp_path,monkeypatch):
    store,project,session=_seed(tmp_path)
    set_splat_updates(tmp_path,session['id'],True,native_orientation_reviewed=True)
    scheduler=Scheduler(tmp_path)
    with patch('bridge.jobs.subprocess.Popen') as launch:
        scheduler.tick_splat()
        launch.assert_not_called()
    runtime=tmp_path/'tools/nerfstudio310-venv/Scripts/python.exe';runtime.parent.mkdir(parents=True);runtime.write_text('fixture')
    for i in range(12): _capture(store,project,session,i)
    class Worker:
        pid=12345678
        def poll(self):return None
    with patch('bridge.jobs.subprocess.Popen',return_value=Worker()) as launch:
        scheduler.tick_splat();scheduler.tick_splat()
        assert launch.call_count==1
        assert '--scheduled' in launch.call_args.args[0]
        assert '--bootstrap' in launch.call_args.args[0]
        assert '--reviewed-native-orientation' in launch.call_args.args[0]
        assert launch.call_args.kwargs['env']['QUEST_SPLAT_COLMAP_EXE']==str(tmp_path/'tools/colmap-4.2.0/bin/colmap.exe')
    set_splat_updates(tmp_path,session['id'],False)
    with patch('bridge.jobs.subprocess.Popen') as launch:
        scheduler.tick_splat();launch.assert_not_called()

def test_splat_status_is_not_depth_capture_count(tmp_path):
    store,project,session=_seed(tmp_path)
    set_splat_updates(tmp_path,session['id'],True,native_orientation_reviewed=True)
    for i in range(3):_capture(store,project,session,i)
    folder=tmp_path/'projects'/project['id']/'online'/session['id'];folder.mkdir(parents=True)
    (folder/'status.json').write_text(json.dumps({'project_id':project['id'],'session_id':session['id'],
      'seed_ids':['a','b'],'accepted_ids':['c','d'],'published_capture_ids':['c'],
      'status':'waiting_for_new_views','artifact':{'vertex_count':321}}))
    status=splat_status(tmp_path,session['id'])
    assert status['trained_views']==3 and status['gaussian_count']==321 and status['total_views']==3
    with _db(tmp_path) as db:
        db.execute('UPDATE splat_updates SET status_json=?',
                   (json.dumps({'status':'blocked','error':'Worker stopped'}),))
    status=splat_status(tmp_path,session['id'])
    assert status['status']=='blocked' and status['error']=='Worker stopped'
    assert status['trained_views']==3


class _YieldingWorker:
    pid = 12345678
    code = None
    def poll(self): return self.code


def _runnable_splat(tmp_path):
    store, project, session = _seed(tmp_path)
    set_splat_updates(tmp_path, session['id'], True, native_orientation_reviewed=True)
    for i in range(12): _capture(store, project, session, i)
    runtime = tmp_path/'tools/nerfstudio310-venv/Scripts/python.exe'
    runtime.parent.mkdir(parents=True, exist_ok=True); runtime.write_text('fixture')
    return store, project, session


def _record_idle(tmp_path, store, project, session, process):
    with _db(tmp_path) as db:
        row = db.execute('SELECT * FROM splat_updates WHERE session_id=?', (session['id'],)).fetchone()
    operation = json.loads(row['status_json'])
    folder = tmp_path/'projects'/project['id']/'online'/session['id']; folder.mkdir(parents=True)
    (folder/'status.json').write_text(json.dumps({'project_id':project['id'], 'session_id':session['id'],
        'status':'waiting_for_new_views', 'worker_state':'idle', 'idle_yield':{
            'input_revision':_depth_inventory(store, session['id'])[2],
            'control_revision':row['control_revision'], 'process_id':process.pid,
            'run_id':operation['run_id']}}))
    process.code = 0
    return folder/'status.json'


@pytest.mark.parametrize('child_pid_offset', [0, 1])
def test_idle_worker_does_not_spin_and_new_inventory_resumes_saved_seed(tmp_path, child_pid_offset):
    store, project, session = _runnable_splat(tmp_path)
    seed = str(uuid.uuid4())
    with _db(tmp_path) as db:
        db.execute('UPDATE splat_updates SET seed_job_id=?', (seed,))
    scheduler, process = Scheduler(tmp_path), _YieldingWorker()
    with patch('bridge.jobs.subprocess.Popen', return_value=process) as launch:
        assert scheduler.tick_splat() is True
        command = launch.call_args.args[0]
        assert command[command.index('--seed-job')+1] == seed and '--bootstrap' not in command
        assert '--run-id' in command and '--control-revision' in command
        # The completed inventory includes captures received after launch.
        _capture(store, project, session, 12)
        path = _record_idle(tmp_path, store, project, session, process)
        receipt = json.loads(path.read_text())
        receipt['idle_yield']['process_id'] += child_pid_offset
        path.write_text(json.dumps(receipt))
        assert scheduler.tick_splat() is False
        for _ in range(3): assert scheduler.tick_splat() is False
        assert launch.call_count == 1
        with _db(tmp_path) as db:
            row = db.execute('SELECT * FROM splat_updates').fetchone()
        assert row['process_id'] is None and row['input_revision'] == _depth_inventory(store, session['id'])[2]
        assert json.loads(row['status_json'])['status'] == 'waiting_for_new_views'
        _capture(store, project, session, 13)
        process.code = None
        assert scheduler.tick_splat() is True and launch.call_count == 2
        assert launch.call_args.args[0][launch.call_args.args[0].index('--seed-job')+1] == seed


def test_idle_worker_releases_gpu_to_another_enabled_project(tmp_path):
    store, project, session = _runnable_splat(tmp_path)
    scheduler, process = Scheduler(tmp_path), _YieldingWorker()
    with patch('bridge.jobs.subprocess.Popen', return_value=process) as launch:
        scheduler.tick_splat()
        other_store, other_project, other_session = _runnable_splat(tmp_path)
        assert scheduler.tick_splat() is True and launch.call_count == 1
        _record_idle(tmp_path, store, project, session, process)
        scheduler.tick_splat()
        assert launch.call_count == 2
        command = launch.call_args.args[0]
        assert command[command.index('--session-id')+1] == other_session['id']


def test_photo_arriving_after_idle_receipt_resumes_without_another_capture(tmp_path):
    store, project, session = _runnable_splat(tmp_path)
    scheduler, process = Scheduler(tmp_path), _YieldingWorker()
    with patch('bridge.jobs.subprocess.Popen', return_value=process) as launch:
        scheduler.tick_splat()
        _record_idle(tmp_path, store, project, session, process)
        _capture(store, project, session, 12)  # Receipt frozen, old process not reconciled yet.
        scheduler.tick_splat()
        assert launch.call_count == 2
        with _db(tmp_path) as db:
            row = db.execute('SELECT * FROM splat_updates').fetchone()
        assert row['input_revision'] == _depth_inventory(store, session['id'])[2]
        assert json.loads(row['status_json'])['status'] == 'running'


def test_manual_final_build_gets_gpu_after_idle_yield(tmp_path):
    store, project, session = _runnable_splat(tmp_path)
    scheduler, process = Scheduler(tmp_path), _YieldingWorker()
    with patch('bridge.jobs.subprocess.Popen', return_value=process) as launch, patch(
        'bridge.jobs.shutil.which', return_value='powershell.exe'):
        scheduler.tick()
        pending = request_build(tmp_path, session['id'], reason='manual', quality='final')
        assert pending['status'] == 'pending'
        scheduler.tick()
        assert launch.call_count == 1  # No competing worker launched against an occupied GPU.
        _record_idle(tmp_path, store, project, session, process)
        scheduler.tick()
        assert launch.call_count == 2
        assert '-Quality' in launch.call_args.args[0] and '--bootstrap' not in launch.call_args.args[0]
        assert launch.call_args.args[0][launch.call_args.args[0].index('-Quality')+1] == 'final'


def test_pause_during_launch_is_retained_and_process_exits_before_bootstrap(tmp_path):
    _store, _project, session = _runnable_splat(tmp_path)
    scheduler, process = Scheduler(tmp_path), _YieldingWorker()
    def launch(command, **_kwargs):
        claimed_control = command[command.index('--control-revision')+1]
        set_splat_updates(tmp_path, session['id'], False)
        with _db(tmp_path) as db:
            changed = db.execute('SELECT control_revision FROM splat_updates').fetchone()[0]
        assert claimed_control != changed
        return process
    with patch('bridge.jobs.subprocess.Popen', side_effect=launch):
        scheduler.tick_splat()
    with _db(tmp_path) as db:
        row = db.execute('SELECT * FROM splat_updates').fetchone()
    assert row['enabled'] == 0 and row['process_id'] == process.pid
    assert json.loads(row['status_json'])['status'] == 'paused'


def test_idle_receipt_is_run_bound_and_graceful_reenable_remains_runnable(tmp_path):
    store, project, session = _runnable_splat(tmp_path)
    scheduler, process = Scheduler(tmp_path), _YieldingWorker()
    with patch('bridge.jobs.subprocess.Popen', return_value=process) as launch:
        scheduler.tick_splat()
        path = _record_idle(tmp_path, store, project, session, process)
        receipt = json.loads(path.read_text()); receipt['idle_yield']['run_id'] = 'another-worker'
        path.write_text(json.dumps(receipt))
        scheduler.tick_splat()
        with _db(tmp_path) as db:
            row = db.execute('SELECT * FROM splat_updates').fetchone()
        assert json.loads(row['status_json'])['status'] == 'blocked'
        # A newer resume command may restart unchanged inventory, but the old
        # worker's graceful control exit must not turn that command into blocked.
        set_splat_updates(tmp_path, session['id'], True)
        receipt.update(status='paused_scheduler', worker_state='stopped')
        path.write_text(json.dumps(receipt))
        with _db(tmp_path) as db:
            db.execute('UPDATE splat_updates SET process_id=?', (process.pid,))
        process.code = 0
        scheduler.tick_splat()
        assert launch.call_count == 2
