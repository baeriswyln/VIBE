using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine.TTS.Inference;
using UnityEngine;
using UnityEngine.Events;

namespace Experiments
{
    /// <summary>
    /// Base class for all experiments. Inherit from this to create new experiment types.
    /// The ExperimentRunner drives experiments through their lifecycle:
    ///   OnBegin() → Run() coroutine → OnEnd()
    /// </summary>
    public abstract class Experiment : MonoBehaviour
    {
        public ExperimentConfig config;
        
        
        [Header("Events")]
        public UnityEvent onExperimentStart;
        public UnityEvent onExperimentComplete;
        public bool skipEndOfExperimentPause = false;

        /// <summary>True while this experiment is actively running.</summary>
        public bool IsRunning { get; private set; }

        /// <summary>True when this experiment has finished.</summary>
        public bool IsComplete { get; private set; }

        [HideInInspector] public bool endScheduled = false;

        protected Experiment(ExperimentConfig config)
        {
            this.config = config;
        }

        /// <summary>
        /// Called by the runner before Run(). Override for one-time setup.
        /// </summary>
        public virtual void OnBegin()
        {
            IsRunning = true;
            IsComplete = false;
            endScheduled = false;
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
            onExperimentComplete?.Invoke();
        }

        /// <summary>
        /// Mark this experiment as finished (call from within Run() or externally).
        /// </summary>
        protected void MarkComplete()
        {
            IsComplete = true;
        }
        
    }
    
    public enum ExperimentType
    {
        Static,
        Random,
    }

    [System.Serializable]
    public class ExperimentConfig
    {
        public ExperimentType type;
        public int numberOfStimuli;
        public int numberOfTrials = 12;
        public bool hasVirtualHand = true;
        public bool hasVisualFeedback = true;
        public bool hasAuditoryFeedback = true;
        public int numberOfErrors = 0;
        public bool pinchIsHinted = false;
        public int numberOfObserveStimuli = 0;
        public bool isChoiceStimuli = false;

        public void CreateExperimentFromSelfAndAddToList(List<Experiment> list)
        {
            switch (type)
            {
                case ExperimentType.Static:
                    list.Add(new ExperimentStatic(this));
                    break;
                case ExperimentType.Random:
                    list.Add(new ExperimentRandom(this));
                    break;
            }
        }
        
    }
}