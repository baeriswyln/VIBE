using System.Collections;
using TMPro;
using Unity.InferenceEngine.TTS.Inference;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Serialization;

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
        public string[] textBeforeStart = new[] {""};

        [TextArea(3, 10)] 
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textBeforeStimuliIntroduction = new[] { "" };
        
        [TextArea(3, 10)] 
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textStimuliIntroduction = new[] { "" };
        
        [TextArea(3, 10)]
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textBeforeExample = new[] {""};
        
        [TextArea(3, 10)]
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textAfterExample = new[] {""};

        public Interactable stimuliPrefab; 
        
        [SerializeField]
        private GazeRaycaster gazeRaycaster;

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

            base.OnEnd();
        }

        public override IEnumerator Run()
        {
            gazeRaycaster.Deactivate();
            if (!trialInfoText)
            {
                Debug.LogError("[TutorialExperiment] No instructionText assigned.");
                yield break;
            }
            
            yield return PlayTextPages(textBeforeStart);
            
            yield return new WaitUntil(() => LslBackend.Instance.startReceived || InputSystem.actions.FindAction("ExpConfirm").IsPressed());
            
            yield return PlayTextPages(textBeforeStimuliIntroduction);

            var baseLoc = new Vector3(-1.5f, 0.5f, 3);
            
            var s_default = Instantiate(stimuliPrefab, baseLoc, Quaternion.identity);
            var s_highlighted = Instantiate(stimuliPrefab, baseLoc + Vector3.right, Quaternion.identity);
            var s_focused = Instantiate(stimuliPrefab, baseLoc + 2*Vector3.right, Quaternion.identity);
            var s_selected = Instantiate(stimuliPrefab, baseLoc + 3*Vector3.right, Quaternion.identity);
            
            s_highlighted.Highlight();
            s_focused.Gaze();
            s_selected.Select();
            
            yield return PlayTextPages(textStimuliIntroduction);
            
            Destroy(s_default.gameObject);
            Destroy(s_highlighted.gameObject);
            Destroy(s_focused.gameObject);
            Destroy(s_selected.gameObject);
            
            var s_example = Instantiate(stimuliPrefab, baseLoc + 1.5f*Vector3.right, Quaternion.identity);
            
            gazeRaycaster.Activate();
            yield return PlayTextPages(textBeforeExample);
            
            // Wait for pinch on stimulus
            yield return new WaitUntil(() => s_example.IsSelected);

            Destroy(s_example.gameObject);
            yield return PlayTextPages(textEnd);

            MarkComplete();
        }

        private void OnPinch()
        {
            _pinchedThisFrame = true;
        }
    }
}