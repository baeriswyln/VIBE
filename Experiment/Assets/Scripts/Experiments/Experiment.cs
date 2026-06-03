using System.Collections;
using TMPro;
using Unity.InferenceEngine.TTS.Inference;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Serialization;

namespace Experiments
{
    /// <summary>
    /// Base class for all experiments. Inherit from this to create new experiment types.
    /// The ExperimentRunner drives experiments through their lifecycle:
    ///   OnBegin() → Run() coroutine → OnEnd()
    /// </summary>
    public abstract class Experiment : MonoBehaviour
    {
        [Header("Experiment Info")]
        [Tooltip("Display name for this experiment (used in logs).")]
        public string experimentName = "Unnamed Experiment";
        
        [TextArea(3, 10)]
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textStart = new[] {""};
        
        [TextArea(3, 10)]
        [Tooltip("Each string becomes one page of instructions.")]
        public string[] textEnd = new[] {""};
        
        [Header("Feedback UI (Optional)")]
        [Tooltip("World-space text for trial info (e.g. 'Trial 3 / 20').")]
        public TMP_Text trialInfoText;

        [Header("Events")]
        public UnityEvent onExperimentStart;
        public UnityEvent onExperimentComplete;
        public bool skipEndOfExperimentPause = false;

        /// <summary>True while this experiment is actively running.</summary>
        public bool IsRunning { get; private set; }

        /// <summary>True when this experiment has finished.</summary>
        public bool IsComplete { get; private set; }

        [HideInInspector] public bool endScheduled = false;

        /// <summary>
        /// Called by the runner before Run(). Override for one-time setup.
        /// </summary>
        public virtual void OnBegin()
        {
            IsRunning = true;
            IsComplete = false;
            endScheduled = false;
            if(trialInfoText) trialInfoText.text = "";
            Debug.Log($"[Experiment] Starting: {experimentName}");
            onExperimentStart?.Invoke();
        }

        /// <summary>
        /// The main experiment logic. Must be implemented by subclasses.
        /// Runs as a coroutine so you can use yield return to wait for things.
        /// </summary>
        public abstract IEnumerator Run();

        /// <summary>
        /// Called by the runner after Run() completes. Override for cleanup.
        /// </summary>
        public virtual void OnEnd()
        {
            IsRunning = false;
            IsComplete = true;
            if(trialInfoText) trialInfoText.text = "";
            Debug.Log($"[Experiment] Completed: {experimentName}");
            onExperimentComplete?.Invoke();
        }

        /// <summary>
        /// Mark this experiment as finished (call from within Run() or externally).
        /// </summary>
        protected void MarkComplete()
        {
            IsComplete = true;
        }
        
        
        protected IEnumerator PlayTextPages(string[] pages)
        {
            if (trialInfoText) trialInfoText.gameObject.SetActive(true);
            for (var i = 0; i < pages.Length && !endScheduled; i++)
            {
                if(trialInfoText) trialInfoText.text = pages[i];
                yield return KokoroTTS.Instance.SpeakAndWait(pages[i]);
            }
        }
    }
}