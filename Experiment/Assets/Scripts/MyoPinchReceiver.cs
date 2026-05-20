using System;
using System.Collections;
using LSL;
using UnityEngine;

public class MyoPinchReceiver : MonoBehaviour
{
    [Header("LSL Stream")] [Tooltip("Stream name as published by the Python bridge.")]
    public string streamName = "MyoPinch";

    [Tooltip("Seconds to wait per resolve attempt before retrying.")]
    public double resolveTimeout = 2.0;

    [Tooltip("How often (s) to retry resolving if the stream isn't up yet.")]
    public float retryInterval = 1.0f;

    [Header("Debug")] public bool logEvents = true;

    /// <summary>Fired on the main thread when a pinch begins.</summary>
    public event Action OnPinchStart;

    /// <summary>Fired on the main thread when a pinch ends.</summary>
    public event Action OnPinchEnd;

    /// <summary>True while the bridge has reported an active pinch.</summary>
    public bool IsPinching { get; private set; }

    /// <summary>True once the LSL stream has been resolved.</summary>
    public bool IsConnected => inlet != null;

    private StreamInlet inlet;
    private string[] sampleBuf = new string[1];

    void OnEnable()
    {
        //StartCoroutine(ResolveLoop());
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
        while (inlet == null && isActiveAndEnabled)
        {
            // resolve_stream blocks the calling thread; LSL's internal timeout
            // keeps it short. We yield between attempts so Unity stays responsive.
            StreamInfo[] results = LSL.LSL.resolve_stream(
                "name", streamName, 1, resolveTimeout);

            if (results != null && results.Length > 0)
            {
                inlet = new StreamInlet(results[0]);
                inlet.open_stream();
                if (logEvents)
                    Debug.Log($"[MyoPinchListener] Connected to '{streamName}'");
                yield break;
            }

            if (logEvents)
                Debug.Log($"[MyoPinchListener] Waiting for '{streamName}'…");
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
            case "pinch_start":
                if (!IsPinching)
                {
                    IsPinching = true;
                    if (logEvents) Debug.Log("[MyoPinchListener] pinch_start");
                    OnPinchStart?.Invoke();
                }

                break;

            case "pinch_end":
                if (IsPinching)
                {
                    IsPinching = false;
                    if (logEvents) Debug.Log("[MyoPinchListener] pinch_end");
                    OnPinchEnd?.Invoke();
                }

                break;

            default:
                if (logEvents)
                    Debug.LogWarning($"[MyoPinchListener] Unknown label: {label}");
                break;
        }
    }
}