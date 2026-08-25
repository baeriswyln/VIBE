using System.Collections;
using Experiments;
using Leap;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;

// Requires the Ultraleap Unity Plugin (com.ultraleap.tracking)

/// <summary>
/// Detects pinch gestures from Ultraleap hand tracking and triggers
/// selection on the currently gazed stimulus.
/// Attach to any GameObject in the scene. Requires a PinchDetector.
/// </summary>
[RequireComponent(typeof(AudioSource))] [DisallowMultipleComponent]
public class PinchHandler : MonoBehaviour
{
    public enum PinchSource {
        Leap,
        Myo,
    }
    
    public static PinchHandler Instance { get; private set; }

    [Header("References")] [Tooltip("The GazeRaycaster that tracks which stimulus is focused.")]
    public GazeRaycaster gazeRaycaster;

    [Tooltip("The PinchDetector in the scene.")]
    public PinchDetector pinchDetector;
    
    [Tooltip("The MyoPinchReceiver in the scene.")]
    public MyoPinchReceiver myoPinchReceiver;

    [Header("Pinch Settings")]
    [Tooltip("Minimum seconds between successive pinch activations (debounce).")]
    public float cooldownTime = 0.2f;

    [Tooltip("Click sound")] public AudioClip clickSound;

    [Tooltip("Select origin of the pinch events")]
    public PinchSource currentPinchSource = PinchSource.Leap;

    [Tooltip("Delay in seconds applied to feedback when feedbackIsDelayed is set on the experiment config.")]
    public float feedbackDelay = 0.5f;

    [Header("Events")] public UnityEvent onPinchDetected;
    public UnityEvent<Interactable> onStimulusClicked;

    private float _lastPinchTime;
    private AudioSource _audioSource;
    public bool hideNextPinch = false;

    public bool Active { get; set; } = true;

    // Singleton initialization
    private void Awake()
    {
        _audioSource = GetComponent<AudioSource>();
        if (_audioSource == null)
            _audioSource = gameObject.AddComponent<AudioSource>();

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
        pinchDetector.OnPinchStart += HandleLeapPinch;
        myoPinchReceiver.OnPinchStart += HandleMyoPinch;
        InputSystem.actions.FindAction("ExpPinch").performed += HandleKeyboardPinch;
    }

    private void OnDisable()
    {
        pinchDetector.OnPinchStart -= HandleLeapPinch;
        myoPinchReceiver.OnPinchStart -= HandleMyoPinch;
        InputSystem.actions.FindAction("ExpPinch").performed -= HandleKeyboardPinch;
    }

    private void HandleKeyboardPinch(InputAction.CallbackContext ctx)
    {
        // No mouse input as pinch if VR is active
        if (ctx.control.device is Mouse && UnityEngine.XR.XRSettings.isDeviceActive)
            return;

        HandlePinch();
    }
    
    private void HandleMyoPinch()
    {
        if(currentPinchSource == PinchSource.Myo) HandlePinch();
    }

    private void HandleLeapPinch(Hand hand)
    {
        if(currentPinchSource == PinchSource.Leap) HandlePinch();
    }
        
    private void HandlePinch()
    {
        if(!Active) return;
        
        // require a minimum time between pinches
        if (_lastPinchTime + cooldownTime > Time.time)
        {
            Debug.LogWarning("pinch skipped");
            return;
        }
        
        _lastPinchTime = Time.time;

            
        // Only select if gaze is confirmed on a target
        if (gazeRaycaster.IsFocusConfirmed && gazeRaycaster.CurrentTarget != null)
        {
            Interactable target = gazeRaycaster.CurrentTarget;
            if (!hideNextPinch)
            {
                if (ExperimentManager.Instance.CurrentExperiment.config.feedbackIsDelayed)
                {
                    if (ExperimentManager.Instance.CurrentExperiment?.config.hasAuditoryFeedback ?? false) 
                        StartCoroutine(PlayClickSoundDelayed(feedbackDelay));
                    
                    StartCoroutine(SelectDelayed(target, feedbackDelay));
                }
                else
                {
                    if (ExperimentManager.Instance.CurrentExperiment?.config.hasAuditoryFeedback ?? false)
                        _audioSource.PlayOneShot(clickSound, .5f);
                    
                    target.OnSelect();
                }
            }
            else
            {
                target.OnIgnoredSelect();
            }
            onStimulusClicked?.Invoke(target);
        }
        else
        {
            onPinchDetected?.Invoke();
        }
    }

    private IEnumerator PlayClickSoundDelayed(float delay)
    {
        yield return new WaitForSeconds(delay);
        _audioSource.PlayOneShot(clickSound, .5f);
    }

    private IEnumerator SelectDelayed(Interactable target, float delay)
    {
        yield return new WaitForSeconds(delay);
        target.OnSelect();
    }
}