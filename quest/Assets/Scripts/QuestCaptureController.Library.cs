using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using GLTFast;
using GLTFast.Loading;
using UnityEngine;
using UnityEngine.Networking;

namespace QuestSplatBridge
{
    public sealed partial class QuestCaptureController
    {
        private const long MaxLibraryGlbBytes = 64L * 1024L * 1024L;
        private readonly List<LibraryChoice> libraryChoices = new List<LibraryChoice>();
        private readonly Dictionary<string, LibraryEntry> libraryEntries = new Dictionary<string, LibraryEntry>();
        private readonly List<LibraryModel> libraryModels = new List<LibraryModel>();
        private const string ProjectsAction = "projects";
        private string libraryFolderId;
        private int librarySelection;
        private bool libraryOpen, libraryBusy;
        private int libraryGeneration;
        private GameObject libraryModelObject;
        private GltfImport libraryModelImport;
        private GltfImport loadingModelImport;
        private CancellationTokenSource loadingModelCancellation;
        private string libraryModelRevision;
        private string libraryModelLabel;

        public bool IsLibraryOpen => libraryOpen;

        // The four physical controller buttons share this entry point.
        public void InvokeControl(int index)
        {
            Debug.Log("Quest Input: button=" + index + "; menu=" + libraryOpen +
                "; HUD=" + hudMode + "; preview=" + requestedPreview);
            // Index 4 is reserved for development automation; the left Menu
            // button calls ExitToHome directly even while model loading.
            if (index == 4) { ExitToHome(); return; }
            if (libraryOpen)
            {
                if (HandleProjectMenuControl(index)) return;
                switch (index)
                {
                    case 0: MoveLibrarySelection(-1); break;
                    case 1: MoveLibrarySelection(1); break;
                    case 2: OpenLibrarySelection(); break;
                    case 3: BackInLibrary(); break;
                }
                return;
            }
            switch (index)
            {
                case 0: RequestCapture(); break;
                case 1: HideLibraryModel(); Preview("splat"); break;
                case 2: ToggleLibraryMesh(); break;
                case 3: OpenProjectsMenu(); break;
            }
        }

        private void OpenLibrary()
        {
            if (!Configured() || !projectVerified)
            {
                status = "Confirm a PC project before browsing models";
                return;
            }
            if (libraryOpen) return;
            ShowFullMenu();
            libraryOpen = true;
            libraryFolderId = null;
            librarySelection = 0;
            RefreshLibraryChoices();
            StartCoroutine(RefreshLibrary());
        }

        private IEnumerator RefreshLibrary()
        {
            if (libraryBusy) yield break;
            libraryBusy = true;
            var generation = ++libraryGeneration;
            var path = "/api/v1/projects/" + UnityWebRequest.EscapeURL(config.project_id);
            foreach (var endpoint in Endpoints())
            {
                string entriesJson = null, modelsJson = null;
                using (var request = UnityWebRequest.Get(endpoint.TrimEnd('/') + path + "/entries"))
                {
                    ConfigureCertificate(request);
                    request.timeout = 8;
                    request.redirectLimit = 0;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return request.SendWebRequest();
                    if (!libraryOpen || generation != libraryGeneration) break;
                    if (request.result == UnityWebRequest.Result.ConnectionError) continue;
                    if (request.result != UnityWebRequest.Result.Success)
                    { status = "Project library unavailable (" + request.responseCode + ")"; break; }
                    entriesJson = request.downloadHandler.text;
                }
                using (var request = UnityWebRequest.Get(endpoint.TrimEnd('/') + path + "/models"))
                {
                    ConfigureCertificate(request);
                    request.timeout = 8;
                    request.redirectLimit = 0;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    yield return request.SendWebRequest();
                    if (!libraryOpen || generation != libraryGeneration) break;
                    if (request.result == UnityWebRequest.Result.ConnectionError) continue;
                    if (request.result != UnityWebRequest.Result.Success)
                    { status = "Model list unavailable (" + request.responseCode + ")"; break; }
                    modelsJson = request.downloadHandler.text;
                }
                try
                {
                    var entries = JsonUtility.FromJson<LibraryEntriesResponse>(entriesJson);
                    var models = JsonUtility.FromJson<LibraryModelsResponse>(modelsJson);
                    if (entries == null || entries.entries == null || models == null || models.models == null)
                        throw new InvalidDataException("Incomplete project library");
                    libraryEntries.Clear();
                    libraryModels.Clear();
                    foreach (var entry in entries.entries)
                        if (entry != null && entry.project_id == config.project_id &&
                            Guid.TryParse(entry.id, out _) &&
                            (entry.kind == "category" || entry.kind == "design" || entry.kind == "variation"))
                            libraryEntries[entry.id] = entry;
                    foreach (var model in models.models)
                        if (ValidLibraryModel(model)) libraryModels.Add(model);
                    RefreshLibraryChoices();
                }
                catch (Exception error)
                {
                    Debug.LogException(error);
                    status = "Project library response invalid";
                }
                break;
            }
            libraryBusy = false;
        }

        private bool ValidLibraryModel(LibraryModel model)
        {
            if (model == null || model.project_id != config.project_id ||
                model.format != "glb" || !Guid.TryParse(model.revision_id, out _) ||
                !Guid.TryParse(model.entry_id, out _) || !libraryEntries.ContainsKey(model.entry_id) ||
                !ValidSha256(model.artifact_sha256) || model.size_bytes <= 0 ||
                model.size_bytes > MaxLibraryGlbBytes ||
                (model.coordinate_frame != "colmap" && model.coordinate_frame != "quest_session") ||
                model.transform == null || model.transform.Length != 16 ||
                model.transform.Any(value => !Finite(value))) return false;
            var expected = "/api/v1/projects/" + config.project_id + "/models/" + model.revision_id + "/artifact";
            return string.Equals(model.artifact_url, expected, StringComparison.Ordinal);
        }

        private void RefreshLibraryChoices()
        {
            libraryChoices.Clear();
            if (libraryFolderId == null)
                libraryChoices.Add(new LibraryChoice { action = ProjectsAction });
            foreach (var entry in libraryEntries.Values)
                if (string.Equals(entry.parent_id, libraryFolderId, StringComparison.Ordinal) ||
                    (libraryFolderId == null && string.IsNullOrEmpty(entry.parent_id)))
                    libraryChoices.Add(new LibraryChoice { entry = entry });
            libraryChoices.Sort((a, b) =>
            {
                if (a.action == ProjectsAction && b.action == ProjectsAction) return 0;
                if (a.action == ProjectsAction) return -1;
                if (b.action == ProjectsAction) return 1;
                var order = a.entry.sort_order.CompareTo(b.entry.sort_order);
                return order != 0 ? order : string.Compare(a.entry.name, b.entry.name, StringComparison.OrdinalIgnoreCase);
            });
            foreach (var model in libraryModels.Where(value => value.entry_id == libraryFolderId)
                         .OrderByDescending(value => value.created_utc))
                libraryChoices.Add(new LibraryChoice { model = model });
            if (librarySelection >= libraryChoices.Count) librarySelection = 0;
            ShowLibraryChoice();
        }

        private void MoveLibrarySelection(int offset)
        {
            if (libraryChoices.Count == 0) return;
            librarySelection = (librarySelection + offset + libraryChoices.Count) % libraryChoices.Count;
            ShowLibraryChoice();
        }

        private void OpenLibrarySelection()
        {
            if (libraryChoices.Count == 0) return;
            var choice = libraryChoices[librarySelection];
            if (choice.action == ProjectsAction)
            {
                OpenProjectsMenu();
                return;
            }
            if (libraryBusy) return;
            if (choice.entry != null)
            {
                libraryFolderId = choice.entry.id;
                librarySelection = 0;
                RefreshLibraryChoices();
            }
            else if (choice.model != null)
                StartCoroutine(DownloadLibraryModel(choice.model));
        }

        private void BackInLibrary()
        {
            if (libraryFolderId == null) { CloseLibrary(); return; }
            if (!libraryEntries.TryGetValue(libraryFolderId, out var current)) libraryFolderId = null;
            else libraryFolderId = string.IsNullOrEmpty(current.parent_id) ? null : current.parent_id;
            librarySelection = 0;
            RefreshLibraryChoices();
        }

        private void ShowLibraryChoice()
        {
            if (!libraryOpen || projectsScreen != ProjectsScreen.Models) return;
            var breadcrumb = new List<string> { CleanLibraryName(projectName) };
            var parent = libraryFolderId;
            var visited = new HashSet<string>();
            while (parent != null && visited.Add(parent) && libraryEntries.TryGetValue(parent, out var entry))
            {
                breadcrumb.Insert(1, CleanLibraryName(entry.name));
                parent = string.IsNullOrEmpty(entry.parent_id) ? null : entry.parent_id;
            }
            var location = string.Join(" / ", breadcrumb);
            if (location.Length > 62) location = "..." + location.Substring(location.Length - 59);
            if (libraryChoices.Count == 0)
            {
                status = location + "\nNo models here; Y goes back";
                return;
            }
            var choice = libraryChoices[librarySelection];
            var name = choice.action == ProjectsAction ? "Projects / New room" :
                choice.entry != null ? choice.entry.name : choice.model.display_name;
            status = location + "\n" + (librarySelection + 1) + "/" + libraryChoices.Count +
                     " " + CleanLibraryName(name) + (choice.action == ProjectsAction || choice.entry != null ? " >" : " [model]");
        }

        private static string CleanLibraryName(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return "Untitled";
            value = Regex.Replace(value, @"\s+", " ").Trim();
            return value.Length > 38 ? value.Substring(0, 35) + "..." : value;
        }

        private void CloseLibrary()
        {
            libraryOpen = false;
            LeaveProjectsMenu();
            ++libraryGeneration;
            status = requestedPreview == "mesh" && libraryModelObject != null ?
                "Inspecting " + CleanLibraryName(libraryModelLabel) + "; capture remains active" :
                "Project confirmed; A captures";
        }

        private void ToggleLibraryMesh()
        {
            if (requestedPreview == "mesh" && libraryModelObject != null)
            {
                HideLibraryModel();
                requestedPreview = null;
                status = "Passthrough; capture remains active";
            }
            else if (libraryModelObject != null)
            {
                if (previewObject != null) previewObject.SetActive(false);
                requestedPreview = "mesh";
                libraryModelObject.SetActive(true);
                UpdateCoverageTint();
                status = "Inspecting " + CleanLibraryName(libraryModelLabel) + "; capture remains active";
            }
            else OpenLibrary();
        }

        public void HideLibraryModel()
        {
            if (libraryModelObject != null) libraryModelObject.SetActive(false);
        }

        public void DisposeLibraryModels()
        {
            ++libraryGeneration;
            if (loadingModelCancellation != null)
            { loadingModelCancellation.Cancel(); loadingModelCancellation.Dispose(); loadingModelCancellation = null; }
            if (loadingModelImport != null) { loadingModelImport.Dispose(); loadingModelImport = null; }
            if (libraryModelObject != null) Destroy(libraryModelObject);
            if (libraryModelImport != null) libraryModelImport.Dispose();
            libraryModelObject = null;
            libraryModelImport = null;
            libraryModelRevision = null;
        }

        private IEnumerator DownloadLibraryModel(LibraryModel model)
        {
            if (libraryBusy) yield break;
            if (model.revision_id == libraryModelRevision && libraryModelObject != null)
            {
                CloseLibrary();
                requestedPreview = null;
                ToggleLibraryMesh();
                yield break;
            }
            libraryBusy = true;
            var generation = ++libraryGeneration;
            status = "Loading " + CleanLibraryName(model.display_name);
            foreach (var endpoint in Endpoints())
            {
                using (var request = UnityWebRequest.Get(endpoint.TrimEnd('/') + model.artifact_url))
                {
                    ConfigureCertificate(request);
                    request.timeout = 60;
                    request.redirectLimit = 0;
                    request.SetRequestHeader("Authorization", "Bearer " + config.token);
                    var operation = request.SendWebRequest();
                    while (!operation.isDone)
                    {
                        if (request.downloadedBytes > (ulong)MaxLibraryGlbBytes || generation != libraryGeneration)
                        { request.Abort(); libraryBusy = false; yield break; }
                        yield return null;
                    }
                    if (request.result == UnityWebRequest.Result.ConnectionError) continue;
                    if (request.result != UnityWebRequest.Result.Success)
                    { status = "Model download failed (" + request.responseCode + ")"; break; }
                    var mime = request.GetResponseHeader("Content-Type");
                    if (mime == null || !mime.StartsWith("model/gltf-binary", StringComparison.OrdinalIgnoreCase))
                    { status = "Model content type rejected"; break; }
                    var bytes = request.downloadHandler.data;
                    if (bytes == null || bytes.LongLength != model.size_bytes ||
                        !string.Equals(Sha256(bytes), model.artifact_sha256, StringComparison.OrdinalIgnoreCase) ||
                        !EmbeddedGlb(bytes))
                    { status = "Model checksum or GLB format rejected"; break; }
                    yield return LoadLibraryGlb(bytes, model, generation);
                    break;
                }
            }
            libraryBusy = false;
        }

        private IEnumerator LoadLibraryGlb(byte[] bytes, LibraryModel model, int generation)
        {
            var candidate = new GameObject("Model preview " + model.revision_id);
            candidate.SetActive(false);
            GltfImport importer = null;
            CancellationTokenSource cancellation = null;
            var success = false;
            string failure = null;
            Task<bool> load = null, instantiate = null;
            GameObjectBoundsInstantiator boundsInstantiator = null;
            try
            {
                cancellation = new CancellationTokenSource();
                importer = new GltfImport(downloadProvider: new DeniedGltfDownloadProvider());
                loadingModelImport = importer;
                loadingModelCancellation = cancellation;
            }
            catch (Exception error)
            {
                failure = error.GetType().Name;
                Debug.LogException(error);
            }
            if (failure == null)
            {
                try { load = importer.LoadGltfBinary(bytes, cancellationToken: cancellation.Token); }
                catch (Exception error) { failure = error.GetType().Name; Debug.LogException(error); }
            }
            if (load != null)
            {
                while (!load.IsCompleted && generation == libraryGeneration) yield return null;
                if (generation != libraryGeneration && !load.IsCompleted)
                {
                    cancellation.Cancel();
                    while (!load.IsCompleted) yield return null;
                }
                if (generation == libraryGeneration && load.Status == TaskStatus.RanToCompletion && load.Result)
                {
                    try
                    {
                        boundsInstantiator = new GameObjectBoundsInstantiator(importer, candidate.transform);
                        instantiate = importer.InstantiateMainSceneAsync(boundsInstantiator, cancellation.Token);
                    }
                    catch (Exception error) { failure = error.GetType().Name; Debug.LogException(error); }
                }
                else failure = "GLB load or selection changed";
            }
            if (instantiate != null)
            {
                while (!instantiate.IsCompleted && generation == libraryGeneration) yield return null;
                if (generation != libraryGeneration && !instantiate.IsCompleted)
                {
                    cancellation.Cancel();
                    while (!instantiate.IsCompleted) yield return null;
                }
                if (generation == libraryGeneration && instantiate.Status == TaskStatus.RanToCompletion && instantiate.Result)
                {
                    var localBounds = boundsInstantiator.CalculateBounds();
                    if (!localBounds.HasValue) failure = "GLB has no renderable mesh";
                    else
                    {
                        var bounds = localBounds.Value;
                        var head = statusCamera.transform;
                        var scale = 1.2f / Mathf.Max(0.1f, bounds.size.magnitude);
                        candidate.transform.localScale = Vector3.one * scale;
                        candidate.transform.position = head.position + head.forward * 1.5f - head.up * 0.18f -
                            candidate.transform.TransformVector(bounds.center);
                        success = true;
                    }
                }
                else failure = "GLB scene load or selection changed";
            }
            loadingModelImport = null;
            loadingModelCancellation = null;
            if (cancellation != null) cancellation.Dispose();
            if (!success)
            {
                if (candidate != null) Destroy(candidate);
                if (importer != null) importer.Dispose();
                if (generation == libraryGeneration) status = "Model load failed: " + failure;
                yield break;
            }
            if (libraryModelObject != null) Destroy(libraryModelObject);
            if (libraryModelImport != null) libraryModelImport.Dispose();
            libraryModelObject = candidate;
            libraryModelImport = importer;
            libraryModelRevision = model.revision_id;
            libraryModelLabel = model.display_name;
            requestedPreview = "mesh";
            if (previewObject != null) previewObject.SetActive(false);
            UpdateCoverageTint();
            candidate.SetActive(true);
            CloseLibrary();
            status = "Inspection model; alignment unverified; capture remains active";
        }

        private static bool EmbeddedGlb(byte[] bytes)
        {
            if (bytes.Length < 20 || BitConverter.ToUInt32(bytes, 0) != 0x46546C67 ||
                BitConverter.ToUInt32(bytes, 4) != 2 || BitConverter.ToUInt32(bytes, 8) != bytes.Length ||
                BitConverter.ToUInt32(bytes, 16) != 0x4E4F534A) return false;
            var jsonLength = BitConverter.ToUInt32(bytes, 12);
            if (jsonLength == 0 || jsonLength > bytes.Length - 20) return false;
            var json = Encoding.UTF8.GetString(bytes, 20, (int)jsonLength);
            // No external buffers, images or extension resources may be fetched.
            return !Regex.IsMatch(json, "\\\"uri\\\"\\s*:", RegexOptions.IgnoreCase);
        }

        [Serializable] private sealed class LibraryEntriesResponse { public LibraryEntry[] entries; }
        [Serializable] private sealed class LibraryModelsResponse { public LibraryModel[] models; }
        [Serializable] private sealed class LibraryEntry
        {
            public string id, project_id, parent_id, kind, name;
            public int sort_order;
        }
        [Serializable] private sealed class LibraryModel
        {
            public string revision_id, project_id, entry_id, format, artifact_sha256;
            public string created_utc, display_name, coordinate_frame, placement_status, artifact_url;
            public long size_bytes;
            public float[] transform;
        }
        private sealed class LibraryChoice
        {
            public string action;
            public LibraryEntry entry;
            public LibraryModel model;
        }

        private sealed class DeniedGltfDownloadProvider : IDownloadProvider
        {
            public Task<IDownload> Request(Uri url) =>
                Task.FromException<IDownload>(new InvalidDataException("External glTF resources are disabled"));

            public Task<ITextureDownload> RequestTexture(Uri url, bool nonReadable) =>
                Task.FromException<ITextureDownload>(new InvalidDataException("External glTF textures are disabled"));
        }
    }
}
