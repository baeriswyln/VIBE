using System.Collections;
using TMPro;
using Unity.InferenceEngine.TTS.Inference;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Experiments
{
    /// <summary>
    /// A simple tutorial experiment that displays instruction text in world space
    /// and advances through a sequence of pages when the user pinches.
    /// </summary>
    public class TutorialExperiment : Experiment
    {
        [Header("Tutorial Pages")]
        [TextArea(3, 10)]
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textBeforeCalibration = new[] {""};

        [TextArea(3, 10)] 
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textBeforeStimuliIntroduction = new[] { "" };
        
        [TextArea(3, 10)] 
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textStimuliIntroduction = new[] { "" };
        
        [TextArea(3, 10)]
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textBeforeExample = new[] {""};

        [Header("UI")]
        [Tooltip("World-space Text (TextMeshPro) that will display the current page.")]
        public TMP_Text instructionText;

        [Tooltip("Optional progress indicator (e.g. '2 / 4').")]
        public TMP_Text progressText;

        public Interactable stimuliPrefab; 
        
        [SerializeField]
        private KokoroTTS kokoroTTS;

        // State
        private bool _pinchedThisFrame;

        public override void OnBegin()
        {
            base.OnBegin();

            PinchHandler.Instance.onPinchDetected.AddListener(OnPinch);
        }

        public override void OnEnd()
        {
            PinchHandler.Instance.onPinchDetected.RemoveListener(OnPinch);

            // Hide the UI
            if (instructionText) instructionText.text = "";
            if (progressText) progressText.text = "";

            base.OnEnd();
        }

        public override IEnumerator Run()
        {
            if (!instructionText)
            {
                Debug.LogError("[TutorialExperiment] No instructionText assigned.");
                yield break;
            }
            
            yield return PlayTextPage(textBeforeCalibration);
            
            // calibration step 1: eye tracking
            Debug.LogWarning("[Calibration] - Perform eye tracking calibration");
            
            // ---------------
            // end calibration
            // ---------------
            
            yield return new WaitUntil(InputSystem.actions.FindAction("ExpConfirm").IsPressed);
            
            yield return PlayTextPage(textBeforeStimuliIntroduction);

            var baseLoc = 3*Vector3.forward + 1.5f*Vector3.left;
            
            var s_default = Instantiate(stimuliPrefab, baseLoc, Quaternion.identity);
            var s_highlighted = Instantiate(stimuliPrefab, baseLoc + Vector3.right, Quaternion.identity);
            var s_focused = Instantiate(stimuliPrefab, baseLoc + 2*Vector3.right, Quaternion.identity);
            var s_selected = Instantiate(stimuliPrefab, baseLoc + 3*Vector3.right, Quaternion.identity);
            
            s_highlighted.Highlight();
            s_focused.Gaze();
            s_selected.Select();
            
            
            Destroy(s_default.gameObject);
            Destroy(s_highlighted.gameObject);
            Destroy(s_focused.gameObject);
            Destroy(s_selected.gameObject);

            yield return PlayTextPage(textStimuliIntroduction);
            
            var s_example = Instantiate(stimuliPrefab, baseLoc + 1.5f*Vector3.right, Quaternion.identity);
            
            yield return PlayTextPage(textBeforeExample);
            
            // todo: deactivate gaze and pinch before here.
            
            // Wait for pinch or auto-advance timeout
            yield return new WaitUntil(() => s_example.IsSelected);

            MarkComplete();
        }

        private IEnumerator PlayTextPage(string[] pages)
        {
            for (var i = 0; i < pages.Length; i++)
            {
                instructionText.text = pages[i];
                yield return kokoroTTS.SpeakAsync(pages[i]);
                yield return kokoroTTS.WaitForEndOfPlayback();

            }
        }

        private void OnPinch()
        {
            _pinchedThisFrame = true;
        }
    }
}