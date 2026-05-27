// ViveOpenXREyeLSLStreamer.cs
// ----------------------------------------------------------------------
// Streams VIVE OpenXR eye tracking data over LSL in two separate streams:
//   1. "ViveOpenXR_Gaze"   (type "Gaze",    continuous)
//   2. "ViveOpenXR_Events" (type "Markers", irregular, string)
//
// Requires:
//   - VIVE.OpenXR + VIVE.OpenXR.EyeTracker (com.htc.upm.vive.openxr)
//   - LSL4Unity (LSL namespace)
//   - "VIVE XR Eye Tracker" feature enabled in OpenXR settings
//
// Drop this on any GameObject. Polls the eye tracker each Update() — the
// OpenXR extension is a polling API, there is no callback equivalent.
// ----------------------------------------------------------------------

using System;
using Logging;
using LSL;
using Newtonsoft.Json;
using UnityEngine;
using VIVE.OpenXR;
using VIVE.OpenXR.EyeTracker;

public class ViveEyeLslStreamer : MonoBehaviour
{
    // -- Inspector knobs ------------------------------------------------
    [Header("Stream identity")]
    [Tooltip("Stable identifier for this device/session. Keep it consistent across runs.")]
    public string sourceId = "ViveOpenXREye_001";

    [Header("Behaviour")]
    [Tooltip("Nominal device rate published in stream metadata. Vive Pro Eye = 120, Focus 3 Eye = 120.")]
    public double nominalRateHz = 120.0;

    [Tooltip("Openness below this counts as 'closed' for blink event detection.")]
    [Range(0f, 1f)] public float blinkOpennessThreshold = 0.15f;
        
    [Tooltip("Number of consecutive 'closed-or-invalid' frames required before a blink_start fires. " +
             "Filters out single-frame tracker hiccups. 2 is a good default at 90+ Hz.")]
    [Range(1, 10)] public int blinkHoldFrames = 2;

    public ExperimentLogger logger;
        
    [Tooltip("If true, samples with no valid gaze data on either eye are skipped instead of pushed as zeros.")]
    public bool skipInvalidSamples = false;

    // -- Stream channel layout -----------------------------------------
    // 19 channels — keep in sync with the metadata block in BuildGazeOutlet().
    private const int N_CHANNELS = 19;
    private static readonly (string label, string eye, string type, string unit)[] CHANNELS =
    {
        ("left_gaze_origin_x",     "left",  "GazeOriginX",    "meters"),
        ("left_gaze_origin_y",     "left",  "GazeOriginY",    "meters"),
        ("left_gaze_origin_z",     "left",  "GazeOriginZ",    "meters"),
        ("left_gaze_direction_x",  "left",  "GazeDirectionX", "normalized"),
        ("left_gaze_direction_y",  "left",  "GazeDirectionY", "normalized"),
        ("left_gaze_direction_z",  "left",  "GazeDirectionZ", "normalized"),
        ("right_gaze_origin_x",    "right", "GazeOriginX",    "meters"),
        ("right_gaze_origin_y",    "right", "GazeOriginY",    "meters"),
        ("right_gaze_origin_z",    "right", "GazeOriginZ",    "meters"),
        ("right_gaze_direction_x", "right", "GazeDirectionX", "normalized"),
        ("right_gaze_direction_y", "right", "GazeDirectionY", "normalized"),
        ("right_gaze_direction_z", "right", "GazeDirectionZ", "normalized"),
        ("left_pupil_diameter",    "left",  "PupilDiameter",  "mm"),
        ("right_pupil_diameter",   "right", "PupilDiameter",  "mm"),
        ("left_eye_openness",      "left",  "EyeOpenness",    "normalized"),
        ("right_eye_openness",     "right", "EyeOpenness",    "normalized"),
        ("left_gaze_valid",        "left",  "Validity",       "boolean"),
        ("right_gaze_valid",       "right", "Validity",       "boolean"),
        ("any_data_valid",         "both",  "Validity",       "boolean"),
    };

    // -- LSL state ------------------------------------------------------
    private StreamOutlet gazeOutlet;
    private StreamOutlet eventOutlet;
    private float[] gazeSample;
    private string[] eventSample;

    // Blink edge detection.
    // We track the *confirmed* (debounced) state plus per-eye streak counters.
    // A "closed-or-invalid" streak of >= blinkHoldFrames flips the state.
    private bool prevLeftClosed  = false;
    private bool prevRightClosed = false;
    private bool prevAnyValid    = true;
    private int  leftClosedStreak  = 0;
    private int  leftOpenStreak    = 0;
    private int  rightClosedStreak = 0;
    private int  rightOpenStreak   = 0;
    private double leftClosedTs  = 0;
    private double leftOpenTs  = 0;
    private double rightClosedTs = 0;
    private double rightOpenTs = 0;
    private bool eyeTrackerAvailable = false;
    private bool warnedUnavailable = false;


    void Start()
    {
        BuildGazeOutlet();
        BuildEventOutlet();
        gazeSample  = new float[N_CHANNELS];
        eventSample = new string[1];
            
        eyeTrackerAvailable = ProbeEyeTracker();
        if (!eyeTrackerAvailable)
        {
            Debug.LogWarning("[ViveEyeLslStreamer] HTC eye tracker extension not available. " +
                             "Gaze sampling disabled; event/marker stream still active. " +
                             "This is normal in the Editor without a headset.");
            PushEvent("eye_tracker_unavailable");
        }
            
        PushEvent("session_start");
    }
        
    private bool ProbeEyeTracker()
    {
        try
        {
            XrSingleEyeGazeDataHTC[] g;
            bool ok = XR_HTC_eye_tracker.Interop.GetEyeGazeData(out g);
            return ok || g != null;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[ViveEyeLslStreamer] Eye tracker probe failed: {e.GetType().Name}: {e.Message}");
            return false;
        }
    }

    void Update()
    {
        if (!eyeTrackerAvailable)
        {
            if (!warnedUnavailable) warnedUnavailable = true; // log once, not every frame
            return;
        }
            
        // OpenXR eye tracker — polled API. Returned data is timestamped
        // by the runtime at the moment of acquisition.
        XrSingleEyeGazeDataHTC[] gazes;
        bool gazeOK = XR_HTC_eye_tracker.Interop.GetEyeGazeData(out gazes);

        XrSingleEyePupilDataHTC[] pupils;
        bool pupilOK = XR_HTC_eye_tracker.Interop.GetEyePupilData(out pupils);

        XrSingleEyeGeometricDataHTC[] geom;
        bool geomOK = XR_HTC_eye_tracker.Interop.GetEyeGeometricData(out geom);

        // Take the LSL timestamp now — the OpenXR call is non-blocking and
        // the returned data was sampled essentially at this Unity frame.
        // (The HTC OpenXR extension does not currently expose a per-sample
        //  hardware timestamp, unlike SRanipal.)
        double lslTs = LSL.LSL.local_clock();

        // Bail early if we got nothing at all.
        if (!gazeOK || gazes == null || gazes.Length < 2)
        {
            // Emit a "data invalid" event on the rising edge.
            if (prevAnyValid) PushEvent("data_invalid", lslTs);
            prevAnyValid = false;
            return;
        }

        // -- Validity flags ---------------------------------------------
        // Index convention from the HTC OpenXR extension:
        //   [0] = left, [1] = right
        bool leftValid  = gazes[0].isValid;
        bool rightValid = gazes[1].isValid;
        bool anyValid   = leftValid || rightValid;

        if (skipInvalidSamples && !anyValid)
        {
            if (prevAnyValid) PushEvent("data_invalid", lslTs);
            prevAnyValid = false;
            return;
        }

        // -- Pack the sample --------------------------------------------
        // Gaze pose: position is in meters in the XR rig's tracking space.
        var lPos = gazes[0].gazePose.position;
        var lOri = gazes[0].gazePose.orientation;
        var rPos = gazes[1].gazePose.position;
        var rOri = gazes[1].gazePose.orientation;

        // OpenXR gaze "direction" is encoded as the orientation's forward
        // (-Z in OpenXR convention). Convert quaternion → forward vector.
        Vector3 lDir = QuatForward(lOri.ToUnityQuaternion());
        Vector3 rDir = QuatForward(rOri.ToUnityQuaternion());

        gazeSample[0]  = lPos.x;
        gazeSample[1]  = lPos.y;
        gazeSample[2]  = lPos.z;
        gazeSample[3]  = lDir.x;
        gazeSample[4]  = lDir.y;
        gazeSample[5]  = lDir.z;
        gazeSample[6]  = rPos.x;
        gazeSample[7]  = rPos.y;
        gazeSample[8]  = rPos.z;
        gazeSample[9]  = rDir.x;
        gazeSample[10] = rDir.y;
        gazeSample[11] = rDir.z;

        // Pupil diameter (mm). Extension reports 0 when invalid.
        float lPupil = (pupilOK && pupils != null && pupils.Length >= 1 && pupils[0].isDiameterValid)
            ? pupils[0].pupilDiameter : 0f;
        float rPupil = (pupilOK && pupils != null && pupils.Length >= 2 && pupils[1].isDiameterValid)
            ? pupils[1].pupilDiameter : 0f;
        gazeSample[12] = lPupil;
        gazeSample[13] = rPupil;

        // Eye openness (0..1). Comes from the geometric data struct.
        float lOpen = (geomOK && geom != null && geom.Length >= 1 && geom[0].isValid)
            ? geom[0].eyeOpenness : 0f;
        float rOpen = (geomOK && geom != null && geom.Length >= 2 && geom[1].isValid)
            ? geom[1].eyeOpenness : 0f;
        gazeSample[14] = lOpen;
        gazeSample[15] = rOpen;

        gazeSample[16] = leftValid  ? 1f : 0f;
        gazeSample[17] = rightValid ? 1f : 0f;
        gazeSample[18] = anyValid   ? 1f : 0f;

        gazeOutlet.push_sample(gazeSample, lslTs);

        // -- Event detection --------------------------------------------
        bool leftRawClosed  = !leftValid  || (lOpen < blinkOpennessThreshold);
        bool rightRawClosed = !rightValid || (rOpen < blinkOpennessThreshold);
        UpdateBlinkState(leftRawClosed, rightRawClosed, lslTs);

        if (anyValid && !prevAnyValid) PushEvent("data_valid",   lslTs);
        if (!anyValid && prevAnyValid) PushEvent("data_invalid", lslTs);
        prevAnyValid = anyValid;
    }
        
    // ------------------------------------------------------------------
    /// <summary>
    /// Debounced blink edge detection. A blink_start fires once an eye has
    /// been continuously "closed-or-invalid" for `blinkHoldFrames` frames;
    /// blink_end fires symmetrically once it's been continuously open for
    /// the same number of frames. This filters single-frame tracker
    /// hiccups without losing real blinks.
    /// </summary>
    private void UpdateBlinkState(bool leftRawClosed, bool rightRawClosed, double lslTs)
    {
        // Left eye
        if (leftRawClosed) { leftClosedStreak++; leftOpenStreak = 0; leftClosedTs = leftClosedStreak == 1 ? lslTs : leftClosedTs; }
        else               { leftOpenStreak++;   leftClosedStreak = 0; leftOpenTs = leftOpenStreak == 1 ? lslTs : leftOpenTs; }
 
        if (!prevLeftClosed && leftClosedStreak >= blinkHoldFrames)
        {
            logger.Log(new BlinkLog("LEFT", "BLINK_START"), leftClosedTs);
            prevLeftClosed = true;
        }
        else if (prevLeftClosed && leftOpenStreak >= blinkHoldFrames)
        {
                
            logger.Log(new BlinkLog("LEFT", "BLINK_END"), leftOpenTs);
            prevLeftClosed = false;
        }
 
        // Right eye
        if (rightRawClosed) { rightClosedStreak++; rightOpenStreak = 0; rightClosedTs = rightClosedStreak == 1 ? lslTs : rightClosedTs; }
        else                { rightOpenStreak++;   rightClosedStreak = 0; rightOpenTs = rightOpenStreak == 1 ? lslTs : rightOpenTs; }
 
        if (!prevRightClosed && rightClosedStreak >= blinkHoldFrames)
        {
            logger.Log(new BlinkLog("RIGHT", "BLINK_START"), rightClosedTs);
            prevRightClosed = true;
        }
        else if (prevRightClosed && rightOpenStreak >= blinkHoldFrames)
        {
            logger.Log(new BlinkLog("RIGHT", "BLINK_END"), rightOpenTs);
            prevRightClosed = false;
        }
    }

        
    // ------------------------------------------------------------------
    /// <summary>
    /// Push an arbitrary string event on the markers stream. Public so
    /// trial / stimulus code can drop markers into the same XDF.
    /// </summary>
    public void PushEvent(string label, double lslTimestamp = 0.0)
    {
        if (eventOutlet == null) return;
        eventSample[0] = label;
        if (lslTimestamp > 0.0) eventOutlet.push_sample(eventSample, lslTimestamp);
        else                    eventOutlet.push_sample(eventSample);
    }

    // ------------------------------------------------------------------
    private static Vector3 QuatForward(Quaternion q)
    {
        // OpenXR poses use -Z forward. In Unity terms this is q * Vector3.forward
        // because Unity's Quaternion already follows the -Z = forward convention
        // for its math, and the OpenXR plugin hands us values consistent with that.
        return q * Vector3.forward;
    }

    // ------------------------------------------------------------------
    private void BuildGazeOutlet()
    {
        var info = new StreamInfo(
            name:           "Experiment.VIBE.ViveOpenXR_Gaze",
            type:           "Gaze",
            channel_count:  N_CHANNELS,
            nominal_srate:  nominalRateHz,
            channel_format: channel_format_t.cf_float32,
            source_id:      sourceId + "_gaze");

        var chns = info.desc().append_child("channels");
        foreach (var (label, eye, type, unit) in CHANNELS)
        {
            var ch = chns.append_child("channel");
            ch.append_child_value("label", label);
            ch.append_child_value("eye", eye);
            ch.append_child_value("type", type);
            ch.append_child_value("unit", unit);
        }

        var acq = info.desc().append_child("acquisition");
        acq.append_child_value("manufacturer", "HTC");
        acq.append_child_value("model", "Vive Pro Eye");
        acq.append_child_value("sdk", "VIVE OpenXR Eye Tracker");

        gazeOutlet = new StreamOutlet(info);
        Debug.Log($"[LSL] Gaze outlet up: {info.name()} ({N_CHANNELS} ch @ {nominalRateHz} Hz)");
    }

    private void BuildEventOutlet()
    {
        var info = new StreamInfo(
            name:           "Experiment.VIBE.ViveOpenXR_Events",
            type:           "Markers",
            channel_count:  1,
            nominal_srate:  0.0,
            channel_format: channel_format_t.cf_string,
            source_id:      sourceId + "_events");

        var acq = info.desc().append_child("acquisition");
        acq.append_child_value("manufacturer", "HTC");
        acq.append_child_value("model", "Vive Pro Eye");

        eventOutlet = new StreamOutlet(info);
        Debug.Log($"[LSL] Events outlet up: {info.name()}");
    }

    void OnDisable()
    {
        PushEvent("session_end");
        gazeOutlet  = null;
        eventOutlet = null;
    }
    
    private class BlinkLog : EventLog
    {
        private string _eye;
        private string _state;
        public BlinkLog(string eye, string state)
        {
            _eye = eye;
            _state = state;
        }

        protected override string GetPayload()
        {
            return JsonConvert.SerializeObject(new
            {
                Eye = _eye,
                State = _state
            });
        }
    }
}