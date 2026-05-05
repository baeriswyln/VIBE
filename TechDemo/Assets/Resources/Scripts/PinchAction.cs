using System;
using Leap;
using UnityEngine;
using UnityEngine.Events;

// Requires the Ultraleap Unity Plugin (com.ultraleap.tracking)

namespace Resources.Scripts
{
    /// <summary>
    /// Detects pinch gestures from Ultraleap hand tracking and triggers
    /// selection on the currently gazed stimulus.
    /// Attach to any GameObject in the scene. Requires a LeapServiceProvider.
    /// </summary>
    public class PinchAction : MonoBehaviour
    {
        [Header("References")] [Tooltip("The GazeRaycaster that tracks which stimulus is focused.")]
        public GazeRaycaster gazeRaycaster;

        [Tooltip("The Leap Pinch Detector in the scene.")]
        public Leap.PinchDetector pinchDetector;

        [Header("Pinch Settings")] [Tooltip("Which hand to track for pinch.")]
        public Chirality handedness = Chirality.Right;

        [Tooltip("Pinch strength threshold (0-1). Higher = more closed pinch required.")] [Range(0f, 1f)]
        public float pinchThreshold = 0.85f;

        [Tooltip("Pinch strength below which the gesture is considered released.")] [Range(0f, 1f)]
        public float releaseThreshold = 0.5f;

        [Tooltip("Minimum seconds between successive pinch activations (debounce).")]
        public float cooldownTime = 0.2f;

        [Tooltip("Click sound")] public AudioClip clickSound;

        [Header("Events")] public UnityEvent onPinchDetected;
        public UnityEvent<Interactable> onStimulusClicked;

        private bool _isPinching;
        private float _lastPinchTime;
        private AudioSource _audioSource;

        public AggregationProviderConfidenceInterpolation apci;

        private void Awake()
        {
            _audioSource = GetComponent<AudioSource>();
            if (_audioSource == null)
                _audioSource = gameObject.AddComponent<AudioSource>();
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
            _audioSource.PlayOneShot(clickSound, .5f);
            Debug.Log("pinch");

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
}