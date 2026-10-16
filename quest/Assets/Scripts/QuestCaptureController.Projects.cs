using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace QuestSplatBridge
{
    public sealed partial class QuestCaptureController
    {
        private enum ProjectsScreen { Models, List, ConfirmCreate, ConfirmSwitch, ConfirmInitialize }
        private const string ProfilesFile = "project-profiles.json";
        private const string PendingRoomFile = "pending-room.json";
        private ProjectsScreen projectsScreen;
        private readonly List<ProjectInfo> visibleProjects = new List<ProjectInfo>();
        private readonly Dictionary<string, ProjectProfile> localProfiles = new Dictionary<string, ProjectProfile>();
        private int projectSelection;
        private int projectMenuGeneration;
        private int configGeneration;
        private bool projectMenuBusy;
        private bool projectSwitching;
        private bool creationNeedsBridgeUpdate;
        private bool creationAttempted;
        private ProjectProfile proposedProfile;
        private ProjectInfo initializeProject;
        private PendingProjectScan pendingProjectScan;
        private string createName;
        private string createProjectRequestId;
        private string createSessionRequestId;
        private string createdProjectId;
        private string createOriginId;
        private Transform initialRigOrigin;
        private Coroutine verificationCoroutine;
        private string projectMenuStatus;
        private int createKind;
        private static readonly string[] ProjectKinds = { "Interior", "Exterior", "Object" };
        private string CurrentMenuStatus => libraryOpen && projectsScreen != ProjectsScreen.Models
            ? projectMenuStatus : status;

        private bool BridgePaired() => config != null && !string.IsNullOrWhiteSpace(config.token) &&
            Uri.TryCreate(config.base_url, UriKind.Absolute, out var uri) && uri.Scheme == Uri.UriSchemeHttps &&
            (string.IsNullOrWhiteSpace(config.wifi_base_url) ||
                (Uri.TryCreate(config.wifi_base_url, UriKind.Absolute, out var wifi) && wifi.Scheme == Uri.UriSchemeHttps)) &&
            (string.IsNullOrEmpty(config.certificate_sha256) || ValidSha256(config.certificate_sha256));

        private void InitializeProjects()
        {
            initialRigOrigin = sessionOrigin;
            var path = Path.Combine(Application.persistentDataPath, ProfilesFile);
            if (File.Exists(path))
            {
                try
                {
                    var catalog = JsonUtility.FromJson<ProjectProfiles>(File.ReadAllText(path));
                    if (catalog?.profiles != null)
                        foreach (var profile in catalog.profiles)
                            if (ValidProfile(profile)) localProfiles[profile.project_id] = profile;
                }
                catch (Exception error) { Debug.LogException(error); }
            }
            if (Configured())
            {
                localProfiles.TryGetValue(config.project_id, out var existing);
                localProfiles[config.project_id] = new ProjectProfile
                {
                    project_id = config.project_id, session_id = config.session_id,
                    tracking_origin_id = config.tracking_origin_id,
                    name = !string.IsNullOrWhiteSpace(projectName) ? projectName :
                        !string.IsNullOrWhiteSpace(existing?.name) ? existing.name : "Current scan",
                    has_capture = existing != null && existing.session_id == config.session_id &&
                        existing.tracking_origin_id == config.tracking_origin_id && existing.has_capture
                };
                TrySaveProfiles();
            }
            var pendingPath = Path.Combine(Application.persistentDataPath, PendingRoomFile);
            if (File.Exists(pendingPath))
            {
                try
                {
                    var pending = JsonUtility.FromJson<PendingRoom>(File.ReadAllText(pendingPath));
                    if (pending != null && !string.IsNullOrWhiteSpace(pending.name) &&
                        Guid.TryParse(pending.project_request_id, out _) &&
                        Guid.TryParse(pending.session_request_id, out _) &&
                        Guid.TryParse(pending.origin_id, out _) &&
                        (string.IsNullOrEmpty(pending.project_id) || Guid.TryParse(pending.project_id, out _)))
                    {
                        createName = pending.name;
                        createProjectRequestId = pending.project_request_id;
                        createSessionRequestId = pending.session_request_id;
                        createOriginId = pending.origin_id;
                        createdProjectId = pending.project_id;
                        creationAttempted = true;
                    }
                }
                catch (Exception error) { Debug.LogException(error); }
            }
        }

        private bool HandleProjectMenuControl(int index)
        {
            if (projectsScreen == ProjectsScreen.Models) return false;
            if (projectsScreen == ProjectsScreen.List)
            {
                if (index == 0 || index == 1)
                {
                    var count = visibleProjects.Count + 2;
                    projectSelection = (projectSelection + (index == 0 ? count - 1 : 1)) % count;
                    ShowProjectChoice();
                }
                else if (index == 2) OpenProjectChoice();
                else if (index == 3) CloseLibrary();
                return true;
            }
            if (projectsScreen == ProjectsScreen.ConfirmCreate && !projectMenuBusy &&
                !creationAttempted && string.IsNullOrEmpty(createdProjectId) && (index == 0 || index == 1))
            {
                createKind = (createKind + (index == 0 ? ProjectKinds.Length - 1 : 1)) % ProjectKinds.Length;
                try { PrepareProjectRequest(); ShowCreateProject(); }
                catch (Exception error) { Debug.LogException(error); projectMenuStatus = "Cannot save project request\nCurrent scan unchanged"; }
                return true;
            }
            if (index == 3)
            {
                // A server-side project may already exist, but the active scan and
                // bridge.json remain unchanged until X confirms a switch.
                ++projectMenuGeneration;
                projectMenuBusy = false;
                projectSwitching = false;
                projectsScreen = ProjectsScreen.List;
                proposedProfile = null;
                initializeProject = null;
                projectMenuStatus = "Projects / New room\nRefreshing PC projects";
                StartCoroutine(RefreshProjects());
            }
            else if (index == 2 && !projectMenuBusy)
            {
                if (projectsScreen == ProjectsScreen.ConfirmCreate)
                    StartCoroutine(CreateRoomOnPc());
                else if (projectsScreen == ProjectsScreen.ConfirmSwitch)
                    StartCoroutine(ConfirmProjectSwitch());
                else if (projectsScreen == ProjectsScreen.ConfirmInitialize)
                    StartCoroutine(InitializeExistingProjectOnPc());
            }
            return true;
        }

        private void OpenProjectsMenu()
        {
            if (!BridgePaired()) { status = "Pair this headset with the PC bridge first"; return; }
            ShowFullMenu();
            libraryOpen = true;
            libraryFolderId = null;
            ++libraryGeneration;
            libraryBusy = false;
            projectsScreen = ProjectsScreen.List;
            projectSelection = 0;
            visibleProjects.Clear();
            projectMenuStatus = "Projects / New room\nLoading projects from PC";
            StartCoroutine(RefreshProjects());
        }

        private void LeaveProjectsMenu()
        {
            if (projectsScreen == ProjectsScreen.Models) return;
            ++projectMenuGeneration;
            projectMenuBusy = false;
            projectSwitching = false;
            projectsScreen = ProjectsScreen.Models;
            proposedProfile = null;
            initializeProject = null;
            if (libraryOpen)
            {
                ShowLibraryChoice();
                StartCoroutine(RefreshLibrary());
            }
        }

        private IEnumerator RefreshProjects()
        {
            var generation = ++projectMenuGeneration;
            foreach (var endpoint in Endpoints())
            {
                using (var request = UnityWebRequest.Get(endpoint.TrimEnd('/') + "/api/v1/projects"))
                {
                    ConfigureCertificate(request);
                    request.timeout = 8;
                    request.redirectLimit = 0;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return request.SendWebRequest();
                    if (generation != projectMenuGeneration || projectsScreen != ProjectsScreen.List) yield break;
                    if (request.result == UnityWebRequest.Result.ConnectionError) continue;
                    if (request.result != UnityWebRequest.Result.Success)
                    { projectMenuStatus = "Projects unavailable (" + request.responseCode + "); Y returns"; yield break; }
                    ProjectList response = null;
                    try { response = JsonUtility.FromJson<ProjectList>(request.downloadHandler.text); }
                    catch (Exception error) { Debug.LogException(error); }
                    if (response?.projects == null)
                    { projectMenuStatus = "Project list invalid; Y returns"; yield break; }
                    visibleProjects.Clear();
                    foreach (var project in response.projects)
                        if (project != null && Guid.TryParse(project.id, out _) &&
                            !string.IsNullOrWhiteSpace(project.name))
                            visibleProjects.Add(project);
                    visibleProjects.Sort((a, b) => string.Compare(a.name, b.name,
                        StringComparison.OrdinalIgnoreCase));
                    if (projectSelection >= visibleProjects.Count + 2) projectSelection = 0;
                    foreach (var project in visibleProjects)
                        if (localProfiles.TryGetValue(project.id, out var profile))
                            profile.name = project.name;
                    yield return RecoverAnchoredProfiles(endpoint, generation);
                    if (generation != projectMenuGeneration || projectsScreen != ProjectsScreen.List) yield break;
                    TrySaveProfiles();
                    ShowProjectChoice();
                    StartCoroutine(LocateProjectAreas(generation));
                    yield break;
                }
            }
            projectMenuStatus = "PC unreachable; Y returns to models";
        }

        private void ShowProjectChoice()
        {
            if (projectsScreen != ProjectsScreen.List) return;
            if (projectSelection == 0)
            {
                projectMenuStatus = "PROJECTS\n> New project\nX Create; A/B Browse\nY Close";
                return;
            }
            if (projectSelection == 1)
            {
                projectMenuStatus = "PROJECTS\n> Models in current project\nX Open; A/B Browse\nY Close";
                return;
            }
            var project = visibleProjects[projectSelection - 2];
            var local = localProfiles.TryGetValue(project.id, out var profile) && ValidProfile(profile);
            projectAreaLabels.TryGetValue(project.id, out var area);
            if (area == "Aligned nearby" && !ProjectAreaResultIsCurrent(project.id)) area = "Check location to open";
            var active = local && project.id == config.project_id && profile.session_id == config.session_id;
            projectMenuStatus = "PROJECTS " + (projectSelection - 1) + "/" + visibleProjects.Count +
                     "\n" + CleanLibraryName(project.name) +
                     "\n" + (active && AnchorReady ? "Current scan here" : area ??
                         (local ? "Checking location" : "No scan saved on headset")) +
                     (local ? "\nX Open; A/B Browse; Y Close" : "\nX Set up first scan; A/B Browse; Y Close");
        }

        private void OpenProjectChoice()
        {
            if (projectMenuBusy) return;
            if (projectSelection == 0)
            {
                // Keep these IDs stable across USB/Wi-Fi retries and a failed
                // second POST. Cancel never changes the active scan.
                if (string.IsNullOrEmpty(createProjectRequestId))
                {
                    try { PrepareProjectRequest(); }
                    catch (Exception error)
                    {
                        Debug.LogException(error);
                        createProjectRequestId = null;
                        projectMenuStatus = "Cannot save room request; current scan unchanged";
                        return;
                    }
                }
                projectsScreen = ProjectsScreen.ConfirmCreate;
                ShowCreateProject();
                return;
            }
            if (projectSelection == 1)
            {
                if (!projectVerified) { projectMenuStatus = "Current project not verified\nCreate or select a project first"; return; }
                LeaveProjectsMenu();
                return;
            }
            var project = visibleProjects[projectSelection - 2];
            if (!localProfiles.TryGetValue(project.id, out var profile) || !ValidProfile(profile))
            {
                try { PrepareExistingProjectScan(project); }
                catch (Exception error)
                {
                    Debug.LogException(error);
                    projectMenuStatus = "Cannot save scan setup request\nCurrent scan unchanged; Y Back";
                    return;
                }
                projectsScreen = ProjectsScreen.ConfirmInitialize;
                projectMenuStatus = "SET UP FIRST SCAN HERE?\n" + CleanLibraryName(project.name) +
                    "\nSame project; saved scans preserved\nX Set up/start; Y Cancel";
                return;
            }
            if (project.id == config.project_id && profile.session_id == config.session_id && AnchorReady)
            { CloseLibrary(); if (requestedPreview != "splat") Preview("splat"); return; }
            proposedProfile = profile;
            projectsScreen = ProjectsScreen.ConfirmSwitch;
            projectMenuStatus = "OPEN PROJECT HERE?\n" + CleanLibraryName(project.name) +
                "\n" + (ProjectNeedsLocation(profile) && !ProjectAreaResultIsCurrent(project.id) ?
                    "Location check required\nX Check/open; Y Cancel" : "X Open; Y Cancel");
        }

        private void PrepareExistingProjectScan(ProjectInfo project)
        {
            if (project == null || !Guid.TryParse(project.id, out _) || string.IsNullOrWhiteSpace(project.name))
                throw new InvalidDataException("Selected project identity is invalid");
            var path = PendingProjectScanPath(project.id);
            PendingProjectScan pending = null;
            if (File.Exists(path)) pending = JsonUtility.FromJson<PendingProjectScan>(File.ReadAllText(path));
            if (pending != null && (pending.project_id != project.id ||
                !Guid.TryParse(pending.session_request_id, out _) || !Guid.TryParse(pending.origin_id, out _)))
                throw new InvalidDataException("Saved scan setup request is invalid");
            if (pending == null)
            {
                pending = new PendingProjectScan { project_id = project.id,
                    session_request_id = Guid.NewGuid().ToString(), origin_id = Guid.NewGuid().ToString() };
                AtomicWrite(path, JsonUtility.ToJson(pending));
            }
            // This stores retry IDs only. No PC write, profile, active session or
            // tracking origin changes until the wearer explicitly confirms X.
            initializeProject = new ProjectInfo { id = project.id, name = project.name };
            pendingProjectScan = pending;
        }

        private static string PendingProjectScanPath(string id) =>
            Path.Combine(Application.persistentDataPath, "pending-project-scan-" + Guid.Parse(id) + ".json");

        private static bool CanInitializeFirstScan(ProjectInfo project, ProjectSessions response, out string reason)
        {
            if (project == null || !Guid.TryParse(project.id, out _) || string.IsNullOrWhiteSpace(project.name) ||
                response?.sessions == null || response.sessions.Any(session => session == null ||
                    session.project_id != project.id || !Guid.TryParse(session.id, out _) ||
                    !Guid.TryParse(session.tracking_origin_id, out _) || session.capture_count < 0 ||
                    string.IsNullOrWhiteSpace(session.status)))
            { reason = "PC scan records unavailable\nCurrent scan unchanged; X Retry, Y Cancel"; return false; }
            if (response.sessions.Any(session => session.capture_count > 0))
            { reason = "Saved photos already exist\nRestore their scan location first\nY Back; current scan unchanged"; return false; }
            reason = null;
            return true;
        }

        private static SessionInfo SelectEmptyFirstScanSession(ProjectSessions response, string retryOrigin) =>
            response.sessions.Where(session => session.capture_count == 0 && session.status == "capturing")
                .OrderByDescending(session => session.tracking_origin_id == retryOrigin)
                .ThenBy(session => session.id, StringComparer.Ordinal).FirstOrDefault();

        private IEnumerator InitializeExistingProjectOnPc()
        {
            if (projectMenuBusy || initializeProject == null || pendingProjectScan == null) yield break;
            if (!CanSwitchProject(out var switchReason)) { projectMenuStatus = switchReason; yield break; }
            var targetProject = initializeProject;
            var pending = pendingProjectScan;
            projectMenuBusy = true;
            creationNeedsBridgeUpdate = false;
            var generation = ++projectMenuGeneration;
            projectMenuStatus = "Checking first scan for\n" + CleanLibraryName(targetProject.name);
            ProjectSessions sessions = null;
            foreach (var endpoint in Endpoints())
            {
                using (var request = UnityWebRequest.Get(endpoint.TrimEnd('/') + "/api/v1/projects/" + targetProject.id))
                {
                    ConfigureCertificate(request); request.timeout = 8; request.redirectLimit = 0;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return request.SendWebRequest();
                    if (generation != projectMenuGeneration) yield break;
                    if (request.result == UnityWebRequest.Result.ConnectionError) continue;
                    ProjectInfo verified = null;
                    try { if (request.result == UnityWebRequest.Result.Success)
                        verified = JsonUtility.FromJson<ProjectInfo>(request.downloadHandler.text); }
                    catch (Exception error) { Debug.LogException(error); }
                    if (verified == null || verified.id != targetProject.id || verified.name != targetProject.name) break;
                }
                using (var request = UnityWebRequest.Get(endpoint.TrimEnd('/') + "/api/v1/projects/" + targetProject.id + "/sessions"))
                {
                    ConfigureCertificate(request); request.timeout = 8; request.redirectLimit = 0;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return request.SendWebRequest();
                    if (generation != projectMenuGeneration) yield break;
                    if (request.result == UnityWebRequest.Result.ConnectionError) continue;
                    try { if (request.result == UnityWebRequest.Result.Success)
                        sessions = JsonUtility.FromJson<ProjectSessions>(request.downloadHandler.text); }
                    catch (Exception error) { Debug.LogException(error); }
                }
                break;
            }
            if (generation != projectMenuGeneration || projectsScreen != ProjectsScreen.ConfirmInitialize) yield break;
            if (!CanInitializeFirstScan(targetProject, sessions, out var reason))
            { projectMenuBusy = false; projectMenuStatus = reason; yield break; }
            var targetSession = SelectEmptyFirstScanSession(sessions, pending.origin_id);
            if (targetSession == null)
            {
                var body = JsonUtility.ToJson(new SessionCreateRequest { project_id = targetProject.id,
                    tracking_origin_id = pending.origin_id, request_id = pending.session_request_id });
                // Deliberately use only the session endpoint: never create or rename a project.
                yield return PostJsonWithFailover("/api/v1/sessions", body, generation, response => {
                    var session = JsonUtility.FromJson<SessionInfo>(response);
                    if (session == null || !Guid.TryParse(session.id, out _) || session.project_id != targetProject.id ||
                        session.tracking_origin_id != pending.origin_id || session.status != "capturing" ||
                        session.capture_count != 0) return false;
                    targetSession = session; return true;
                });
            }
            if (generation != projectMenuGeneration || projectsScreen != ProjectsScreen.ConfirmInitialize) yield break;
            projectMenuBusy = false;
            if (targetSession == null)
            {
                projectMenuStatus = creationNeedsBridgeUpdate ? "Update PC bridge for safe setup\nCurrent scan unchanged" :
                    "Scan setup failed; X Retry, Y Cancel\nCurrent scan unchanged";
                yield break;
            }
            if (!CanSwitchProject(out switchReason)) { projectMenuStatus = switchReason; yield break; }
            var profile = new ProjectProfile { project_id = targetProject.id, name = targetProject.name,
                session_id = targetSession.id, tracking_origin_id = targetSession.tracking_origin_id };
            if (localProfiles.TryGetValue(targetProject.id, out var prior) && ValidProfile(prior))
            { projectMenuStatus = "Scan profile now exists\nY Back, then reopen this project"; yield break; }
            localProfiles[targetProject.id] = profile;
            try { SaveProfiles(); }
            catch (Exception error)
            {
                localProfiles.Remove(targetProject.id);
                Debug.LogException(error); projectMenuStatus = "Local setup cannot be saved\nX Retry, Y Cancel"; yield break;
            }
            // Keeping the tiny durable retry record is intentional: interrupted
            // setup and later reopening reuse the same IDs without touching old data.
            proposedProfile = profile;
            projectsScreen = ProjectsScreen.ConfirmSwitch;
            Debug.Log("Quest Splat: first-scan profile prepared for existing project " + targetProject.id +
                "; session " + targetSession.id + "; no project created");
            yield return ConfirmProjectSwitch();
        }

        private IEnumerator CreateRoomOnPc()
        {
            if (projectMenuBusy) yield break;
            projectMenuBusy = true;
            creationAttempted = true;
            creationNeedsBridgeUpdate = false;
            var generation = ++projectMenuGeneration;
            projectMenuStatus = "Creating " + createName + " on PC";
            if (string.IsNullOrEmpty(createdProjectId))
            {
                var body = JsonUtility.ToJson(new ProjectCreateRequest
                { name = createName, request_id = createProjectRequestId });
                yield return PostJsonWithFailover("/api/v1/projects", body, generation,
                    response =>
                    {
                        var project = JsonUtility.FromJson<ProjectInfo>(response);
                        if (project != null && Guid.TryParse(project.id, out _) && project.name == createName)
                        { createdProjectId = project.id; return true; }
                        return false;
                    });
            }
            if (generation != projectMenuGeneration || projectsScreen != ProjectsScreen.ConfirmCreate)
                yield break;
            if (string.IsNullOrEmpty(createdProjectId))
            {
                projectMenuBusy = false;
                projectMenuStatus = creationNeedsBridgeUpdate ? "Update PC bridge for safe creation" :
                    "PC creation failed; X retries, Y cancels";
                yield break;
            }
            try { SavePendingRoom(); }
            catch (Exception error)
            {
                Debug.LogException(error);
                projectMenuBusy = false;
                projectMenuStatus = "Room request cannot be saved; X retries";
                yield break;
            }
            string createdSessionId = null;
            var sessionBody = JsonUtility.ToJson(new SessionCreateRequest
            { project_id = createdProjectId, tracking_origin_id = createOriginId,
                request_id = createSessionRequestId });
            yield return PostJsonWithFailover("/api/v1/sessions", sessionBody, generation,
                response =>
                {
                    var session = JsonUtility.FromJson<SessionInfo>(response);
                    if (session != null && Guid.TryParse(session.id, out _) &&
                        session.project_id == createdProjectId &&
                        session.tracking_origin_id == createOriginId && session.status == "capturing")
                    { createdSessionId = session.id; return true; }
                    return false;
                });
            if (generation != projectMenuGeneration || projectsScreen != ProjectsScreen.ConfirmCreate)
                yield break;
            projectMenuBusy = false;
            if (string.IsNullOrEmpty(createdSessionId))
            {
                projectMenuStatus = creationNeedsBridgeUpdate ? "Update PC bridge for safe creation" :
                    "Room session failed; X retries, Y cancels";
                yield break;
            }
            proposedProfile = new ProjectProfile
            {
                project_id = createdProjectId, session_id = createdSessionId,
                tracking_origin_id = createOriginId, name = createName
            };
            // This is a saved destination, not a change of active project.
            localProfiles[proposedProfile.project_id] = proposedProfile;
            try
            {
                SaveProfiles();
                var pendingPath = Path.Combine(Application.persistentDataPath, PendingRoomFile);
                if (File.Exists(pendingPath)) File.Delete(pendingPath);
                createProjectRequestId = null;
                createSessionRequestId = null;
                createdProjectId = null;
                createOriginId = null;
                creationAttempted = false;
            }
            catch (Exception error)
            {
                Debug.LogException(error);
                projectMenuStatus = "Room made; local record failed; X retries";
                yield break;
            }
            projectsScreen = ProjectsScreen.ConfirmSwitch;
            projectMenuStatus = "Created " + createName + "\nStarting fresh scan here";
            yield return ConfirmProjectSwitch();
        }

        private IEnumerator PostJsonWithFailover(string path, string json, int generation,
            Func<string, bool> accept)
        {
            foreach (var endpoint in Endpoints())
            {
                using (var health = UnityWebRequest.Get(endpoint.TrimEnd('/') + "/health"))
                {
                    ConfigureCertificate(health);
                    health.timeout = 5;
                    health.redirectLimit = 0;
                    health.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return health.SendWebRequest();
                    if (generation != projectMenuGeneration) yield break;
                    if (health.result != UnityWebRequest.Result.Success) continue;
                    BridgeHealth capabilities = null;
                    try { capabilities = JsonUtility.FromJson<BridgeHealth>(health.downloadHandler.text); }
                    catch (Exception error) { Debug.LogException(error); }
                    if (capabilities?.capabilities == null ||
                        !capabilities.capabilities.Contains("idempotent_creation"))
                    {
                        creationNeedsBridgeUpdate = true;
                        continue;
                    }
                }
                using (var request = new UnityWebRequest(endpoint.TrimEnd('/') + path, "POST"))
                {
                    request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
                    request.downloadHandler = new DownloadHandlerBuffer();
                    request.SetRequestHeader("Content-Type", "application/json");
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    request.timeout = 10;
                    request.redirectLimit = 0;
                    ConfigureCertificate(request);
                    yield return request.SendWebRequest();
                    if (generation != projectMenuGeneration) yield break;
                    if (request.result == UnityWebRequest.Result.ConnectionError ||
                        request.responseCode >= 500) continue;
                    if (request.result == UnityWebRequest.Result.Success)
                    {
                        var accepted = false;
                        try { accepted = accept(request.downloadHandler.text); }
                        catch (Exception error) { Debug.LogException(error); }
                        if (accepted) yield break;
                        continue;
                    }
                    yield break;
                }
            }
        }

        private IEnumerator ConfirmProjectSwitch()
        {
            if (projectMenuBusy || proposedProfile == null || !ValidProfile(proposedProfile)) yield break;
            if (!CanSwitchProject(out var reason)) { projectMenuStatus = reason; yield break; }
            projectMenuBusy = true;
            projectSwitching = true;
            var generation = ++projectMenuGeneration;
            var target = proposedProfile;
            projectMenuStatus = "Verifying " + CleanLibraryName(target.name) + " on PC";
            var verified = false;
            foreach (var endpoint in Endpoints())
            {
                var baseUrl = endpoint.TrimEnd('/') + "/api/v1/";
                using (var request = UnityWebRequest.Get(baseUrl + "sessions/" + target.session_id))
                {
                    ConfigureCertificate(request);
                    request.timeout = 8;
                    request.redirectLimit = 0;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return request.SendWebRequest();
                    if (generation != projectMenuGeneration) yield break;
                    if (request.result == UnityWebRequest.Result.ConnectionError) continue;
                    if (request.result != UnityWebRequest.Result.Success) break;
                    SessionInfo session = null;
                    try { session = JsonUtility.FromJson<SessionInfo>(request.downloadHandler.text); }
                    catch (Exception error) { Debug.LogException(error); }
                    verified = session != null && session.id == target.session_id &&
                        session.project_id == target.project_id &&
                        session.tracking_origin_id == target.tracking_origin_id &&
                        session.status == "capturing";
                    if (verified) target.has_capture |= session.capture_count > 0;
                }
                break;
            }
            if (generation != projectMenuGeneration) yield break;
            string laterReason = null;
            if (!verified || !CanSwitchProject(out laterReason))
            {
                projectSwitching = false;
                projectMenuBusy = false;
                projectMenuStatus = verified ? laterReason : "Target scan not verified; still in current project";
                yield break;
            }
            if (ProjectNeedsLocation(target))
            {
                projectMenuStatus = "Locating " + CleanLibraryName(target.name) + "\nLook around briefly";
                yield return LocateProjectAreas(generation);
                if (generation != projectMenuGeneration) yield break;
                if (!ProjectAreaResultIsCurrent(target.project_id))
                { projectSwitching = false; projectMenuBusy = false; projectMenuStatus = "Location no longer tracked\nCurrent project unchanged"; yield break; }
            }
            var next = new BridgeConfig
            {
                base_url = config.base_url, wifi_base_url = config.wifi_base_url,
                token = config.token, certificate_sha256 = config.certificate_sha256,
                project_id = target.project_id, session_id = target.session_id,
                tracking_origin_id = target.tracking_origin_id
            };
            try
            {
                if (Configured() && (!localProfiles.TryGetValue(config.project_id, out var previousProfile) ||
                    !previousProfile.has_capture || sessionHasCapture))
                    localProfiles[config.project_id] = new ProjectProfile
                    { project_id = config.project_id, session_id = config.session_id,
                      tracking_origin_id = config.tracking_origin_id, name = projectName,
                      has_capture = sessionHasCapture };
                localProfiles[target.project_id] = target;
                SaveProfiles();
                AtomicWrite(Path.Combine(Application.persistentDataPath, ConfigFile), JsonUtility.ToJson(next));
            }
            catch (Exception error)
            {
                Debug.LogException(error);
                projectSwitching = false;
                projectMenuBusy = false;
                projectMenuStatus = "Could not save switch; current scan unchanged";
                yield break;
            }
            ++configGeneration;
            if (verificationCoroutine != null) StopCoroutine(verificationCoroutine);
            config = next;
            projectName = target.name;
            projectVerified = false;
            capturePending = false;
            sessionHasCapture = false;
            originInvalidated = true;
            requestedPreview = null;
            previewAligned = false;
            displayMode = "passthrough";
            DisposePreview();
            DisposeLibraryModels();
            libraryEntries.Clear(); libraryModels.Clear(); libraryChoices.Clear();
            libraryBusy = false;
            libraryOpen = false;
            projectsScreen = ProjectsScreen.Models;
            ++projectMenuGeneration;
            if (sessionAnchor != null) Destroy(sessionAnchor.gameObject);
            sessionAnchor = null;
            sessionOrigin = initialRigOrigin;
            anchorSaved = false;
            anchorReadyAnnounced = false;
            anchorStableSince = -1f;
            LoadSessionState();
            verificationCoroutine = StartCoroutine(VerifyProjectLoop());
            projectSwitching = false;
            projectMenuBusy = false;
            status = "Switching to " + CleanLibraryName(target.name) + "; locating scan";
            if (target.has_capture) StartCoroutine(OpenLocatedProjectPreview(configGeneration));
        }

        private bool ProjectNeedsLocation(ProjectProfile profile) => profile.has_capture ||
            File.Exists(Path.Combine(Application.persistentDataPath, "anchor-" + profile.session_id + ".json"));

        private void PrepareProjectRequest()
        {
            createName = ProjectKinds[createKind] + " " + DateTime.Now.ToString("MMM d h:mm tt",
                System.Globalization.CultureInfo.InvariantCulture);
            createProjectRequestId = Guid.NewGuid().ToString();
            createSessionRequestId = Guid.NewGuid().ToString();
            createOriginId = Guid.NewGuid().ToString();
            createdProjectId = null;
            SavePendingRoom();
        }

        private void ShowCreateProject() => projectMenuStatus =
            (string.IsNullOrEmpty(createdProjectId) ? "NEW PROJECT" : "RESUME CREATION") +
            "\n" + createName + (creationAttempted ? "\nX Retry/start; Y Cancel" : "\nA/B Type; X Create/start\nY Cancel");

        private IEnumerator OpenLocatedProjectPreview(int generation)
        {
            var readyBy = Time.unscaledTime + 25;
            while (generation == configGeneration && Time.unscaledTime < readyBy && (!projectVerified || !AnchorReady))
                yield return null;
            if (generation == configGeneration && projectVerified && AnchorReady && !libraryOpen)
                Preview("splat");
        }

        private void OrderProjectsByLocation()
        {
            var selectedId = projectSelection >= 2 && projectSelection < visibleProjects.Count + 2
                ? visibleProjects[projectSelection - 2].id : null;
            visibleProjects.Sort((a, b) => {
                var nearby = projectsInArea.Contains(b.id).CompareTo(projectsInArea.Contains(a.id));
                return nearby != 0 ? nearby : string.Compare(a.name, b.name, StringComparison.OrdinalIgnoreCase);
            });
            if (selectedId != null) projectSelection = visibleProjects.FindIndex(p => p.id == selectedId) + 2;
        }

        private IEnumerator RecoverAnchoredProfiles(string endpoint, int generation)
        {
            // An earlier anchored scan can outlive a later empty pairing. Use
            // authoritative PC identity/counts; never infer project IDs from filenames.
            if (Directory.GetFiles(Application.persistentDataPath, "anchor-*.json").Length == 0) yield break;
            foreach (var project in visibleProjects)
            {
                if (localProfiles.TryGetValue(project.id, out var existing) && existing.has_capture &&
                    File.Exists(Path.Combine(Application.persistentDataPath, "anchor-" + existing.session_id + ".json"))) continue;
                using (var request = UnityWebRequest.Get(endpoint.TrimEnd('/') + "/api/v1/projects/" + project.id + "/sessions"))
                {
                    ConfigureCertificate(request); request.timeout = 5; request.redirectLimit = 0;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return request.SendWebRequest();
                    if (generation != projectMenuGeneration) yield break;
                    if (request.result != UnityWebRequest.Result.Success) continue;
                    ProjectSessions response = null;
                    try { response = JsonUtility.FromJson<ProjectSessions>(request.downloadHandler.text); }
                    catch (Exception error) { Debug.LogException(error); }
                    if (response?.sessions == null) continue;
                    foreach (var session in response.sessions.OrderByDescending(s => s.capture_count))
                    {
                        if (session.project_id != project.id || session.status != "capturing" || session.capture_count <= 0 ||
                            !Guid.TryParse(session.id, out _) || !Guid.TryParse(session.tracking_origin_id, out _)) continue;
                        var anchorPath = Path.Combine(Application.persistentDataPath, "anchor-" + session.id + ".json");
                        if (!File.Exists(anchorPath)) continue;
                        AnchorRecord record = null;
                        try { record = JsonUtility.FromJson<AnchorRecord>(File.ReadAllText(anchorPath)); }
                        catch (Exception error) { Debug.LogException(error); }
                        if (record == null || record.session_id != session.id || record.origin_id != session.tracking_origin_id ||
                            !Guid.TryParse(record.anchor_id, out _)) continue;
                        localProfiles[project.id] = new ProjectProfile { project_id = project.id, session_id = session.id,
                            tracking_origin_id = session.tracking_origin_id, name = project.name, has_capture = true };
                        break;
                    }
                }
            }
        }

        private bool CanSwitchProject(out string reason)
        {
            if (capturePending || uploadRunning ||
                Directory.EnumerateFileSystemEntries(queuePath).Any())
            { reason = "Wait for all saved photos to upload before switching"; return false; }
            if (exitRequested)
            { reason = "App is exiting"; return false; }
            reason = null;
            return true;
        }

        private void TrySaveProfiles()
        {
            try { SaveProfiles(); }
            catch (Exception error) { Debug.LogException(error); }
        }

        private void RememberActiveCaptureState()
        {
            if (!Configured()) return;
            localProfiles[config.project_id] = new ProjectProfile
            {
                project_id = config.project_id, session_id = config.session_id,
                tracking_origin_id = config.tracking_origin_id,
                name = string.IsNullOrWhiteSpace(projectName) ? "Current scan" : projectName,
                has_capture = sessionHasCapture
            };
            TrySaveProfiles();
        }

        private void SaveProfiles()
        {
            var json = JsonUtility.ToJson(new ProjectProfiles
            { profiles = localProfiles.Values.Where(ValidProfile).ToArray() });
            AtomicWrite(Path.Combine(Application.persistentDataPath, ProfilesFile), json);
        }

        private void SavePendingRoom()
        {
            AtomicWrite(Path.Combine(Application.persistentDataPath, PendingRoomFile),
                JsonUtility.ToJson(new PendingRoom
                {
                    name = createName, project_request_id = createProjectRequestId,
                    session_request_id = createSessionRequestId,
                    origin_id = createOriginId, project_id = createdProjectId
                }));
        }

        private static void AtomicWrite(string path, string contents)
        {
            var temporary = path + ".tmp";
            File.WriteAllText(temporary, contents);
            try
            {
                if (File.Exists(path)) File.Replace(temporary, path, null);
                else File.Move(temporary, path);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }

        private static bool ValidProfile(ProjectProfile profile) => profile != null &&
            Guid.TryParse(profile.project_id, out _) &&
            Guid.TryParse(profile.session_id, out _) &&
            Guid.TryParse(profile.tracking_origin_id, out _);

        [Serializable] private sealed class PendingProjectScan
        { public string project_id, session_request_id, origin_id; }
        [Serializable] private sealed class ProjectProfiles { public ProjectProfile[] profiles; }
        [Serializable] private sealed class PendingRoom
        {
            public string name, project_request_id, session_request_id, origin_id, project_id;
        }
        [Serializable] private sealed class ProjectProfile
        { public string project_id, session_id, tracking_origin_id, name; public bool has_capture; }
        [Serializable] private sealed class ProjectList { public ProjectInfo[] projects; }
        [Serializable] private sealed class ProjectSessions { public SessionInfo[] sessions; }
        [Serializable] private sealed class ProjectCreateRequest { public string name, request_id; }
        [Serializable] private sealed class BridgeHealth
        { public string status, protocol_version; public string[] capabilities; }
        [Serializable] private sealed class SessionCreateRequest
        { public string project_id, tracking_origin_id, request_id; }
    }
}
