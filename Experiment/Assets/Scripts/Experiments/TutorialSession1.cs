using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Experiments
{
    /// <summary>
    /// A simple tutorial experiment that displays instruction text in world space
    /// and advances through a sequence of pages when the user pinches.
    /// </summary>
    public class TutorialSession1 : Experiment
    {
        [Header("Tutorial Pages")] [TextArea(3, 10)] [Tooltip("Each string becomes one page of instructions.")]
        public string[] textBeforeStart = new[] { "" };

        [TextArea(3, 10)] [Tooltip("Each string becomes one page of instructions.")]
        public string[] textBeforeStimuliIntroduction = new[] { "" };

        [TextArea(3, 10)] [Tooltip("Each string becomes one page of instructions.")]
        public string[] textStimuliIntroduction = new[] { "" };

        [TextArea(3, 10)] [Tooltip("Each string becomes one page of instructions.")]
        public string[] textBeforeExample = new[] { "" };

        [TextArea(3, 10)] [Tooltip("Each string becomes one page of instructions.")]
        public string[] textAfterExample = new[] { "" };

        public Interactable stimuliPrefab;

        [SerializeField] private GazeRaycaster gazeRaycaster;

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

            yield return ExperimentManager.Instance.PlayTextPages(textBeforeStart, false);

            yield return new WaitUntil(() =>
                LslBackend.Instance.startReceived || InputSystem.actions.FindAction("ExpConfirm").IsPressed() ||
                endScheduled);

            yield return ExperimentManager.Instance.PlayTextPages(textBeforeStimuliIntroduction);

            var baseLoc = new Vector3(-1.5f, 0.5f, 3);

            var s_default = Instantiate(stimuliPrefab, baseLoc, Quaternion.identity);
            var s_highlighted = Instantiate(stimuliPrefab, baseLoc + Vector3.right, Quaternion.identity);
            var s_focused = Instantiate(stimuliPrefab, baseLoc + 2 * Vector3.right, Quaternion.identity);
            var s_selected = Instantiate(stimuliPrefab, baseLoc + 3 * Vector3.right, Quaternion.identity);

            s_highlighted.Highlight();
            s_focused.Gaze();
            s_selected.Select();

            // yield return ExperimentManager.Instance.PlayTextPages(textStimuliIntroduction);

            Destroy(s_default.gameObject);
            Destroy(s_highlighted.gameObject);
            Destroy(s_focused.gameObject);
            Destroy(s_selected.gameObject);

            gazeRaycaster.Activate();
            PinchHandler.Instance.Active = true;
            // 4 rounds of Static_4 normal
            var cfg1 = new ExperimentConfig(ExperimentType.Static, 4, 4, numberOfErrors:4);
            yield return RunTutorialBlock(cfg1, "TutorialSession1_Block1");

            // 4 rounds of Static_4 with "observe"
            var cfg2 = new ExperimentConfig(ExperimentType.Static, 4, 4, numberOfObserveStimuli: 2);
            yield return RunTutorialBlock(cfg2, "TutorialSession1_Block2");

            // 4 rounds of Static_4 with "observe" and "hinted pinch"
            var cfg3 = new ExperimentConfig(ExperimentType.Static, 4, 4, numberOfObserveStimuli: 2,
                pinchIsHinted: true);
            yield return RunTutorialBlock(cfg3, "TutorialSession1_Block3");

            // any questions? wait for confirmation by 
            yield return new WaitUntil(() => InputSystem.actions.FindAction("ExpConfirm").IsPressed() || endScheduled);
            PinchHandler.Instance.Active = false;

            yield return ExperimentManager.Instance.PlayTextPages(textAfterExample);

            MarkComplete();
        }

        // todo: possibiltiy that experiment navigation does not work while in a sub-experiment
        private IEnumerator RunTutorialBlock(ExperimentConfig cfg, string name)
        {
            var comp = cfg.CreateExperiment(ExperimentManager.Instance.gameObject, name);

            comp.OnBegin();
            yield return comp.Run();
            comp.OnEnd();
            
            Destroy(comp.gameObject);
        }

        private void OnPinch()
        {
        }
    }
}