using System.Collections;
using UnityEngine;
using UnityEngine.Events;

namespace Resources.Scripts.Experiments
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

        [Header("Events")]
        public UnityEvent onExperimentStart;
        public UnityEvent onExperimentComplete;

        public ExperimentManager em;

        /// <summary>True while this experiment is actively running.</summary>
        public bool IsRunning { get; private set; }

        /// <summary>True when this experiment has finished.</summary>
        public bool IsComplete { get; private set; }

        /// <summary>
        /// Called by the runner before Run(). Override for one-time setup.
        /// </summary>
        public virtual void OnBegin()
        {
            IsRunning = true;
            IsComplete = false;
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
    }
}