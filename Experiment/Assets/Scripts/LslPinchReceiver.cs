using System;
using System.Collections;
using System.Threading.Tasks;
using Experiments;
using LSL;
using UnityEngine;

public class LslPinchReceiver : MonoBehaviour
{
    [Header("LSL Stream")] [Tooltip("Stream name as published by the Python bridge.")]
    public string streamName = "Experiment.VIBE.Pinch";

    [Tooltip("Seconds to wait per resolve attempt before retrying.")]
    public double resolveTimeout = 0;

    [Tooltip("How often (s) to retry resolving if the stream isn't up yet.")]
    public float retryInterval = 1.0f;

    [Tooltip("Minimum time between accepted pinches")]
    public float pinchDebounceTime = 1f;

    [Header("Debug")] public bool logEvents = true;

    /// <summary>Fired on the main thread when a pinch begins.</summary>
    public event Action OnPinch;

    /// <summary>True once the LSL stream has been resolved.</summary>
    public bool IsConnected => inlet != null;

    private StreamInlet inlet;
    private string[] sampleBuf = new string[1];
    private float tsLastPinch;

    void OnEnable()
    {
        tsLastPinch = Time.time;
        StartCoroutine(ResolveLoop());
    }

    void OnDisable()
    {
        if (inlet != null)
        {
            inlet.Close();
            inlet = null;
        }
    }

    private IEnumerator ResolveLoop()
    {
        while (inlet == null && isActiveAndEnabled && !(ExperimentManager.Instance?.IsRunning ?? false))
        {
            // resolve_stream blocks the calling thread -> delegate to thread
            Task<StreamInfo[]> resolveTask = Task.Run(() =>
                LSL.LSL.resolve_stream("name", streamName, 1, resolveTimeout));
            
            yield return new WaitUntil(() => resolveTask.IsCompleted);
            
            StreamInfo[] results = resolveTask.Result;
            if (results is { Length: > 0 })
            {
                inlet = new StreamInlet(results[0]);
                inlet.open_stream();
                if (logEvents)
                    Debug.Log($"[PinchListener] Connected to '{streamName}'");
                yield break;
            }

            if (logEvents)
                Debug.Log($"[PinchListener] Waiting for '{streamName}'…");
            yield return new WaitForSeconds(retryInterval);
        }
    }

    void Update()
    {
        if (inlet == null) return;

        // Drain everything that arrived since the last frame. pull_sample
        // with timeout=0 is non-blocking and returns 0 when the queue is empty.
        // Cap the loop to avoid a runaway frame if something pathological happens.
        for (int i = 0; i < 64; i++)
        {
            double ts = inlet.pull_sample(sampleBuf, 0.0);
            if (ts == 0.0) break;
            Dispatch(sampleBuf[0]);
        }
    }

    private void Dispatch(string label)
    {
        switch (label)
        {
            case "PINCH":
                // debounce
                var now = Time.time;
                if (now - tsLastPinch >  pinchDebounceTime)
                {
                    tsLastPinch = now;
                    OnPinch?.Invoke();
                }

                break;

            default:
                if (logEvents)
                    Debug.LogWarning($"[PinchListener] Unknown label: {label}");
                break;
        }
    }
}