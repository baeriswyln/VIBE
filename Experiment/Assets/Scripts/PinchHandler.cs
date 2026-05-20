using Leap;
using UnityEngine;
using UnityEngine.Events;

// Requires the Ultraleap Unity Plugin (com.ultraleap.tracking)

/// <summary>
/// Detects pinch gestures from Ultraleap hand tracking and triggers
/// selection on the currently gazed stimulus.
/// Attach to any GameObject in the scene. Requires a PinchDetector.
/// </summary>
[RequireComponent(typeof(AudioSource))] [DisallowMultipleComponent]
public class PinchHandler : MonoBehaviour
{
    public static PinchHandler Instance { get; private set; }

    [Header("References")] [Tooltip("The GazeRaycaster that tracks which stimulus is focused.")]
    public GazeRaycaster gazeRaycaster;

    [Tooltip("The PinchDetector in the scene.")]
    public PinchDetector pinchDetector;

    [Header("Pinch Settings")]
    [Tooltip("Minimum seconds between successive pinch activations (debounce).")]
    public float cooldownTime = 0.2f;

    [Tooltip("Click sound")] public AudioClip clickSound;

    [Header("Events")] public UnityEvent onPinchDetected;
    public UnityEvent<Interactable> onStimulusClicked;

    private float _lastPinchTime;
    private AudioSource _audioSource;
        
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
        pinchDetector.OnPinchStart += HandlePinch;
    }

    private void OnDisable()
    {
        pinchDetector.OnPinchStart -= HandlePinch;
    }
        
    private void HandlePinch(Hand hand)
    {
        // require a minimum time between pinches
        if (_lastPinchTime + cooldownTime > Time.time)
        {
            Debug.LogWarning("pinch accepted");
            return;
        }

        _audioSource.PlayOneShot(clickSound, .5f);
        _lastPinchTime = Time.time;

            
        // Only select if gaze is confirmed on a target
        if (gazeRaycaster.IsFocusConfirmed && gazeRaycaster.CurrentTarget != null)
        {
            Interactable target = gazeRaycaster.CurrentTarget;
            target.OnSelect();
            onStimulusClicked?.Invoke(target);
        }
        else
        {
            onPinchDetected?.Invoke();
        }
    }
}