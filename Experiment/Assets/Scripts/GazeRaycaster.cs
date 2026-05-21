// VIVE OpenXR Plugin native eye tracker API
// Requires: VIVE OpenXR Plugin with "Vive XR Eye Tracker (Beta)" enabled
// in Project Settings > XR Plug-in Management > OpenXR > OpenXR Feature Groups

using Logging;
using UnityEngine;
using UnityEngine.Events;
using VIVE.OpenXR;
using VIVE.OpenXR.EyeTracker;

/// <summary>
/// Casts a ray using eye tracking data from the HTC Vive Pro Eye.
///
/// Uses three gaze sources in priority order:
///   1. VIVE OpenXR native Eye Tracker API  (XR_HTC_eye_tracker — most reliable)
///   2. OpenXR Eye Gaze Interaction profile (Input System — generic/cross-platform)
///   3. Head gaze fallback                  (camera forward — always works)
///
/// The VIVE native API (option 1) is the recommended path for the Vive Pro Eye.
/// The generic Input System path often returns (0,0,1) on this headset because
/// the standard XR_EXT_eye_gaze_interaction extension isn't always forwarded
/// correctly through SteamVR's OpenXR runtime.
///
/// Setup:
///   - Install VIVE OpenXR Plugin (2.5+) and Unity OpenXR Plugin.
///   - In Project Settings > XR Plug-in Management > OpenXR:
///       • Enable "Vive XR Eye Tracker (Beta)" in the feature groups.
///   - Attach this script to your Main Camera (XR Origin camera).
/// </summary>
public class GazeRaycaster : MonoBehaviour
{
    [Header("Gaze Source")]
    [Tooltip("Use eye tracking when available. If false, always uses head gaze.")]
    public bool useEyeTracking = true;

    [Tooltip("Falls back to head-gaze when eye tracking data is invalid.")]
    public bool fallbackToHeadGaze = true;

    [Tooltip("Which eye to use for the gaze ray.")]
    public GazeEye gazeEye = GazeEye.Combined;
        
    [Tooltip("Number of consecutive 'closed-or-invalid' frames required before a blink_start fires. " +
             "Filters out single-frame tracker hiccups. 2 is a good default at 90+ Hz.")]
    [Range(1, 10)] public int blinkHoldFrames = 2;


    [Header("Raycast Settings")] [Tooltip("Maximum distance for gaze detection.")]
    public float maxDistance = 20f;

    [Tooltip("Layer mask for stimulus objects.")]
    public LayerMask stimulusLayer = ~0;

    [Tooltip("Radius of the spherecast (> 0 recommended for eye tracking noise).")]
    public float gazeRadius = 0.03f;

    [Header("Focus Timing")]
    [Tooltip("Seconds the user must look at a stimulus before it counts as focused.")]
    public float focusDwellTime = 0.0f;

    [Header("Visual Feedback")]
    [Tooltip("Optional reticle transform to position at the gaze hit point.")]
    public Transform reticle;

    [Tooltip("Default distance for the reticle when nothing is hit.")]
    public float reticleDefaultDistance = 3f;

    [Header("Smoothing")]
    [Tooltip("Smooth the eye gaze direction to reduce jitter. 0 = none, higher = more.")]
    [Range(0f, 20f)]
    public float gazeSmoothingFactor = 5f;

    [Header("Debug")] 
    [Tooltip("Draw gaze ray in Scene view.")]
    public bool drawDebugRay = true;
        
    [Tooltip("Width of the visible debug ray.")]
    public float debugRayWidth = 0.003f;

    [Tooltip("Log gaze source and raw values to the console.")]
    public bool logDebug = false;
        
    // Runtime-created LineRenderer for in-headset debug visualization
    private LineRenderer _debugLine;

    [Header("Events")]
    public UnityEvent<Interactable> onStimulusFocused;
    public UnityEvent<Interactable> onStimulusUnfocused;

    // Enums
    public enum GazeEye { Combined, Left, Right }
    public enum GazeSource { None, ViveNativeEyeTracker, OpenXRInputSystem, HeadGaze }

    // State
    private Interactable _currentTarget;
    private float _focusTimer;
    private Vector3 _smoothedGazeDirection;
    private GazeSource _activeSource = GazeSource.None;
    private GazeSource _prevSource = GazeSource.None;
    private bool _isActive = true;

    // --- Public API ---

    public Interactable CurrentTarget => _currentTarget;
    public bool IsFocusConfirmed => _currentTarget != null && _focusTimer >= focusDwellTime;

    public bool IsEyeTrackingActive => _activeSource == GazeSource.ViveNativeEyeTracker
                                       || _activeSource == GazeSource.OpenXRInputSystem;

    public Vector3 CurrentGazeDirection => _smoothedGazeDirection;
    public Vector3 CurrentGazeOrigin { get; private set; }
    public GazeSource ActiveSource => _activeSource;

    // -------------------------------------------------------------------

    private void Start()
    {
        _smoothedGazeDirection = transform.forward;
    }

    private void Update()
    {
        if (!_isActive) return;
        
        Vector3 rayOrigin;
        Vector3 rayDirection;
        bool gotEyeTracking = false;

        if (useEyeTracking)
        {
            // Priority 1: VIVE native eye tracker API
            gotEyeTracking = TryGetViveNativeEyeGaze(out rayOrigin, out rayDirection);

            // Priority 2: Head gaze fallback
            if (!gotEyeTracking)
            {
                if (fallbackToHeadGaze)
                {
                    rayOrigin = transform.position;
                    rayDirection = transform.forward;
                    _activeSource = GazeSource.HeadGaze;
                }
                else
                {
                    _activeSource = GazeSource.None;
                    return;
                }
            }
        }
        else
        {
            rayOrigin = transform.position;
            rayDirection = transform.forward;
            _activeSource = GazeSource.HeadGaze;
        }

        // Log source transitions
        if (logDebug && _activeSource != _prevSource)
        {
            Debug.Log($"[GazeRaycaster] Source changed: {_prevSource} → {_activeSource}");
            _prevSource = _activeSource;
        }

        // Smooth gaze direction
        if (gazeSmoothingFactor > 0f)
        {
            _smoothedGazeDirection = Vector3.Lerp(
                _smoothedGazeDirection,
                rayDirection,
                Time.deltaTime * gazeSmoothingFactor
            ).normalized;
        }
        else
        {
            _smoothedGazeDirection = rayDirection;
        }

        CurrentGazeOrigin = rayOrigin;

        // Raycast
        Ray gazeRay = new Ray(rayOrigin, _smoothedGazeDirection);
        RaycastHit hit;
        bool didHit = gazeRadius > 0f
            ? Physics.SphereCast(gazeRay, gazeRadius, out hit, maxDistance, stimulusLayer)
            : Physics.Raycast(gazeRay, out hit, maxDistance, stimulusLayer);

        Interactable hitStimulus = null;
        if (didHit)
        {
            hitStimulus = hit.collider.GetComponentInParent<Interactable>();
            UpdateReticle(hit.point, hit.normal);
        }
        else
        {
            UpdateReticle(
                rayOrigin + _smoothedGazeDirection * reticleDefaultDistance,
                -_smoothedGazeDirection
            );
        }

        // Focus transitions
        if (hitStimulus != _currentTarget)
        {
            if (_currentTarget != null)
            {
                _currentTarget.OnGazeExit();
                onStimulusUnfocused?.Invoke(_currentTarget);
            }

            _currentTarget = hitStimulus;
            _focusTimer = 0f;

            if (_currentTarget != null)
            {
                _currentTarget.OnGazeEnter();
                onStimulusFocused?.Invoke(_currentTarget);
            }
        }

        if (_currentTarget != null)
            _focusTimer += Time.deltaTime;

        // Debug ray
        if (drawDebugRay)
        {
            // Debug ray — visible in headset via LineRenderer
            UpdateDebugRay(rayOrigin, didHit ? hit.point : rayOrigin + _smoothedGazeDirection * maxDistance);
        }
    }

    public void Activate()
    {
        _isActive = true;
        reticle.gameObject.SetActive(true);
    }

    public void Deactivate()
    {
        _isActive = false;
        reticle.gameObject.SetActive(false);
    }

    private void UpdateDebugRay(Vector3 start, Vector3 end)
    {
        if (!drawDebugRay)
        {
            if (_debugLine != null) _debugLine.enabled = false;
            return;
        }

        // Create LineRenderer on demand
        if (_debugLine == null)
        {
            var go = new GameObject("GazeDebugRay");
            go.transform.SetParent(transform, false);
            _debugLine = go.AddComponent<LineRenderer>();
            _debugLine.positionCount = 2;
            _debugLine.startWidth = debugRayWidth;
            _debugLine.endWidth = debugRayWidth;
            _debugLine.material = new Material(Shader.Find("Sprites/Default"));
            _debugLine.useWorldSpace = true;
            _debugLine.receiveShadows = false;
            _debugLine.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        }

        _debugLine.enabled = true;

        Color c = _activeSource switch
        {
            GazeSource.ViveNativeEyeTracker => Color.green,
            GazeSource.OpenXRInputSystem => Color.yellow,
            _ => Color.cyan
        };
        _debugLine.startColor = c;
        _debugLine.endColor = c;
        _debugLine.SetPosition(0, start);
        _debugLine.SetPosition(1, end);
    }

    // =======================================================================
    // PRIORITY 1: VIVE OpenXR native Eye Tracker API (XR_HTC_eye_tracker)
    // This is the most reliable path for the Vive Pro Eye.
    // The gaze pose is returned in head-relative space.
    // =======================================================================
    private bool TryGetViveNativeEyeGaze(out Vector3 origin, out Vector3 direction)
    {
        origin = Vector3.zero;
        direction = Vector3.forward;

        try
        {
            bool success = XR_HTC_eye_tracker.Interop.GetEyeGazeData(
                out XrSingleEyeGazeDataHTC[] gazes
            );

            if (!success || gazes == null || gazes.Length == 0)
                return false;


            // Pick the requested eye
            XrSingleEyeGazeDataHTC gazeData;
            switch (gazeEye)
            {
                case GazeEye.Left:
                    gazeData = gazes[(int)XrEyePositionHTC.XR_EYE_POSITION_LEFT_HTC];
                    break;
                case GazeEye.Right:
                    gazeData = gazes[(int)XrEyePositionHTC.XR_EYE_POSITION_RIGHT_HTC];
                    break;
                default: // Combined
                    // Use left eye as base; average with right if both valid
                    var left = gazes[(int)XrEyePositionHTC.XR_EYE_POSITION_LEFT_HTC];
                    var right = gazes[(int)XrEyePositionHTC.XR_EYE_POSITION_RIGHT_HTC];
                    if (left.isValid && right.isValid)
                    {
                        // Average the two gaze directions for a combined ray
                        Vector3 leftDir = (left.gazePose.orientation.ToUnityQuaternion()) * Vector3.forward;
                        Vector3 rightDir = (right.gazePose.orientation.ToUnityQuaternion()) * Vector3.forward;
                        Vector3 leftPos = (left.gazePose.position.ToUnityVector());
                        Vector3 rightPos = (right.gazePose.position.ToUnityVector());

                        origin = (leftPos + rightPos) * 0.5f;
                        direction = ((leftDir + rightDir) * 0.5f).normalized;

                        _activeSource = GazeSource.ViveNativeEyeTracker;

                        if (logDebug)
                            Debug.Log($"[GazeRaycaster] VIVE native (combined): dir={direction}");

                        return true;
                    }

                    // Fall through to whichever single eye is valid
                    gazeData = left.isValid ? left : right;
                    break;
            }

            if (!gazeData.isValid)
                return false;

            // Transform from head-relative space to world space
            Vector3 localPos = gazeData.gazePose.position.ToUnityVector();
            Quaternion localRot = gazeData.gazePose.orientation.ToUnityQuaternion();

            origin = transform.TransformPoint(localPos);
            direction = (transform.rotation * localRot * Vector3.forward).normalized;

            _activeSource = GazeSource.ViveNativeEyeTracker;

            if (logDebug)
                Debug.Log($"[GazeRaycaster] VIVE native ({gazeEye}): dir={direction}");

            return true;
        }
        catch (System.Exception e)
        {
            if (logDebug)
                Debug.LogWarning($"[GazeRaycaster] VIVE native API error: {e.Message}");
            return false;
        }
    }

    private void UpdateReticle(Vector3 position, Vector3 normal)
    {
        if (reticle == null) return;
        reticle.position = position;
        reticle.rotation = Quaternion.LookRotation(normal);
    }

    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.cyan;
        Gizmos.DrawRay(transform.position, transform.forward * maxDistance);
    }
}