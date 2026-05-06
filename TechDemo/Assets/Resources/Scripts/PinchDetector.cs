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
    public class PinchDetector : MonoBehaviour
    {
        public static PinchDetector Instance { get; private set; }

        [Header("References")] [Tooltip("The GazeRaycaster that tracks which stimulus is focused.")]
        public GazeRaycaster gazeRaycaster;

        [Tooltip("The Leap Service Provider in the scene.")]
        public LeapServiceProvider leapProvider;

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
            apci.OnFixedFrame += ProcessFrame;
        }

        private void OnDisable()
        {
            apci.OnFixedFrame -= ProcessFrame;
        }

        private void ProcessFrame(Frame frame)
        {
            if (leapProvider == null || gazeRaycaster == null) return;

            Hand hand = GetTrackedHand(frame);
            if (hand == null)
            {
                _isPinching = false;
                return;
            }

            float pinchStrength = hand.PinchStrength;

            // Detect pinch onset (transition from open to pinched)
            if (!_isPinching && pinchStrength >= pinchThreshold)
            {
                _isPinching = true;

                if (Time.time - _lastPinchTime < cooldownTime)
                    return;

                _lastPinchTime = Time.time;
                HandlePinch();
            }
            // Detect release
            else if (_isPinching && pinchStrength < releaseThreshold)
            {
                _isPinching = false;
            }
        }

        private void HandlePinch()
        {
            _audioSource.PlayOneShot(clickSound, .5f);

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

        private Hand GetTrackedHand(Frame frame)
        {
            // Frame frame = leapProvider.CurrentFrame;
            if (frame == null) return null;

            foreach (Hand hand in frame.Hands)
            {
                if (handedness == Chirality.Right && hand.IsRight) return hand;
                if (handedness == Chirality.Left && hand.IsLeft) return hand;
            }

            return null;
        }
    }
}