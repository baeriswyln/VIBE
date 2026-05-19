using System.Collections;
using TMPro;
using Unity.InferenceEngine.Samples.TTS.Inference;
using UnityEngine;

namespace Resources.Scripts.Experiments
{
    /// <summary>
    /// A simple tutorial experiment that displays instruction text in world space
    /// and advances through a sequence of pages when the user pinches.
    /// </summary>
    public class TutorialExperiment : Experiment
    {
        [Header("Tutorial Pages")]
        [TextArea(3, 10)]
        [Tooltip("Each string becomes one page of instructions. User pinches to advance.")]
        public string[] pages = new[]
        {
            "Welcome!\n\nIn this experiment you will select visual stimuli using your gaze and a pinch gesture.",
            "Look at a stimulus to highlight it.\nIt will change color when you focus on it.",
            "To select a stimulus, pinch your thumb and index finger together while looking at it.",
            "Try it now: pinch to continue."
        };

        [Header("UI")]
        [Tooltip("World-space Text (TextMeshPro) that will display the current page.")]
        public TMP_Text instructionText;

        [Tooltip("Optional progress indicator (e.g. '2 / 4').")]
        public TMP_Text progressText;

        [Header("Advance Trigger")]
        [Tooltip("Minimum seconds a page must be shown before a pinch can advance it (prevents accidental skipping).")]
        public float minTimePerPage = 1.0f;

        [Header("Auto-advance (Optional)")]
        [Tooltip("If > 0, automatically advance after this many seconds (in addition to pinch).")]
        public float autoAdvanceAfterSeconds = 0f;
        
        [SerializeField]
        private KokoroTTS kokoroTTS;

        // State
        private bool _pinchedThisFrame;

        public override void OnBegin()
        {
            base.OnBegin();

            PinchDetector.Instance.onPinchDetected.AddListener(OnPinch);
        }

        public override void OnEnd()
        {
            PinchDetector.Instance.onPinchDetected.RemoveListener(OnPinch);

            // Hide the UI
            if (instructionText != null) instructionText.text = "";
            if (progressText != null) progressText.text = "";

            base.OnEnd();
        }

        public override IEnumerator Run()
        {
            if (instructionText == null)
            {
                Debug.LogError("[TutorialExperiment] No instructionText assigned.");
                yield break;
            }

            for (int i = 0; i < pages.Length; i++)
            {
                instructionText.text = pages[i];
                _ = kokoroTTS.SpeakAsync(MisakiSharp.TextToPhonemes(pages[i]));

                if (progressText != null)
                    progressText.text = $"{i + 1} / {pages.Length}";

                // Wait the minimum time before accepting input
                yield return new WaitForSeconds(minTimePerPage);

                // Wait for pinch or auto-advance timeout
                _pinchedThisFrame = false;
                float pageStartTime = Time.time;

                while (true)
                {
                    if (_pinchedThisFrame)
                        break;

                    if (autoAdvanceAfterSeconds > 0f &&
                        Time.time - pageStartTime >= autoAdvanceAfterSeconds)
                        break;

                    yield return null;
                }

                // Brief pause for visual feedback
                yield return new WaitForSeconds(0.2f);
            }

            MarkComplete();
        }

        private void OnPinch()
        {
            _pinchedThisFrame = true;
        }
    }
}