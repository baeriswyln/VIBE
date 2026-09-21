using System.Collections;
using System.Threading.Tasks;
using LSL;
using UnityEngine;

public class LslBackend : MonoBehaviour
{
    public static LslBackend Instance { get; private set; }

    // -- Inspector knobs ------------------------------------------------
    [Header("LSL Stream")] [Tooltip("Stable identifier for this device/session. Keep it consistent across runs.")]
    public string sourceId = "UnityBackend";

    [Tooltip("Backend orchestrator stream name.")]
    public string inletStreamName = "Experiment.VIBE.Backend.Orchestrator";

    [Tooltip("Backend game stream name.")]
    public string outletStreamName = "Experiment.VIBE.Backend.Game";

    [Tooltip("Seconds to wait per resolve attempt before retrying.")]
    public double resolveTimeout = 2.0;

    [Tooltip("How often (s) to retry resolving if the stream isn't up yet.")]
    public float retryInterval = 1.0f;

    // -- LSL state ------------------------------------------------------
    private StreamInlet backendInlet;
    private StreamOutlet backendOutlet;

    public bool startReceived { get; private set; } = false;
    private string[] eventSample = new string[1];
    private string[] inletBuf = new string[1];

    void Start()
    {
        BuildBackendOutlet();
    }

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);
    }


    private void OnEnable()
    {
        StartCoroutine(BuildBackendInlet());
    }

    void Update()
    {
        if (backendInlet != null)
        {
            for (int i = 0; i < 64; i++)
            {
                double ts = backendInlet.pull_sample(inletBuf, 0.0);
                if (ts == 0.0) break;
                Dispatch(inletBuf[0]);
            }
        }
    }

    void OnDisable()
    {
        if (backendInlet != null)
        {
            backendInlet.Close();
            backendInlet = null;
        }

        backendOutlet = null;
    }


    // ------------------------------------------------------------------
    /// <summary>
    /// Push an arbitrary string event on the markers stream. Public so
    /// trial / stimulus code can drop markers into the same XDF.
    /// </summary>
    public void PushEvent(string label, double lslTimestamp = 0.0)
    {
        if (backendOutlet == null) return;
        eventSample[0] = label;
        if (lslTimestamp > 0.0) backendOutlet.push_sample(eventSample, lslTimestamp);
        else backendOutlet.push_sample(eventSample);
    }

    private void Dispatch(string label)
    {
        switch (label)
        {
            case "game_start":
                startReceived = true;
                PushEvent("game_start_ack");
                break;
            case "shutdown":
                Quit();
                break;
        }
    }

    private void Quit()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
    Application.Quit();
#endif
    }

    private void BuildBackendOutlet()
    {
        var info = new StreamInfo(
            name: outletStreamName,
            type: "Markers",
            channel_count: 1,
            nominal_srate: 0.0,
            channel_format: channel_format_t.cf_string,
            source_id: sourceId + "_events");

        backendOutlet = new StreamOutlet(info);
        Debug.Log($"[LSL] Events outlet up: {info.name()}");
    }

    private IEnumerator BuildBackendInlet()
    {
        while (backendInlet == null && isActiveAndEnabled)
        {
            // resolve_stream blocks the calling thread -> delegate to thread
            Task<StreamInfo[]> resolveTask = Task.Run(() =>
                LSL.LSL.resolve_stream("name", inletStreamName, 1, resolveTimeout));
            
            yield return new WaitUntil(() => resolveTask.IsCompleted);
            
            StreamInfo[] results = resolveTask.Result;
            if (results is { Length: > 0 })
            {
                backendInlet = new StreamInlet(results[0]);
                backendInlet.open_stream();

                Debug.Log($"[LSL] Connected to '{inletStreamName}'");
                yield break;
            }

            Debug.Log($"[LSL] Waiting for '{inletStreamName}'…");
            yield return new WaitForSeconds(retryInterval);
        }
    }
}