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

    [Header("Events")] public UnityEvent onPinchDetected;
    public UnityEvent<Interactable> onStimulusClicked;

    private float _lastPinchTime;
    private AudioSource _audioSource;
    public bool hideNextPinch;

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
        InputSystem.actions.FindAction("ExpPause").performed += HandleKeyboardPinch;
    }

    private void OnDisable()
    {
        pinchDetector.OnPinchStart -= HandleLeapPinch;
        myoPinchReceiver.OnPinchStart -= HandleMyoPinch;
        InputSystem.actions.FindAction("ExpPause").performed -= HandleKeyboardPinch;
    }

    private void HandleKeyboardPinch(InputAction.CallbackContext ctx)
    {
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

        // don't play sound if the pinch should be hidden, or no auditory feedback is required.
        if (!hideNextPinch && ExperimentManager.Instance.CurrentExperiment.config.hasAuditoryFeedback) 
            _audioSource.PlayOneShot(clickSound, .5f);
        
        _lastPinchTime = Time.time;

            
        // Only select if gaze is confirmed on a target
        if (gazeRaycaster.IsFocusConfirmed && gazeRaycaster.CurrentTarget != null)
        {
            Interactable target = gazeRaycaster.CurrentTarget;
            if (!hideNextPinch)
            {
                target.OnSelect();
                onStimulusClicked?.Invoke(target);
            }
            else
            {
                // todo
                Debug.LogError("the hidden pinches are not logged yet");
            }
            hideNextPinch = false;
        }
        else
        {
            onPinchDetected?.Invoke();
        }
    }
}