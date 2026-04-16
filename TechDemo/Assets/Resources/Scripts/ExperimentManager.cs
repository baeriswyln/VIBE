using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Events;
using Debug = UnityEngine.Debug;

namespace Resources.Scripts
{
    public class ExperimentManager : MonoBehaviour
    {
        [Header("Experiment Sequence")] [Tooltip("Experiments will run in this order.")]
        public List<Experiment> experiments = new List<Experiment>();

        [Header("Start Behaviour")]
        [Tooltip("Start the sequence automatically on Play. If false, call StartSequence() manually.")]
        public bool startOnPlay = true;

        [Tooltip("Seconds to wait between experiments.")]
        public float gapBetweenExperiments = 1f;

        [Header("Events")] public UnityEvent onSequenceStart;
        public UnityEvent onSequenceComplete;
        public UnityEvent<Experiment> onExperimentStarted;
        public UnityEvent<Experiment> onExperimentFinished;

        // State
        private int _currentIndex = -1;
        private bool _isRunning;

        DateTime sessionStartUtc = DateTime.UtcNow;
        Stopwatch stopwatch = Stopwatch.StartNew();

        public bool IsRunning => _isRunning;
        public int CurrentIndex => _currentIndex;

        public Experiment CurrentExperiment =>
            (_currentIndex >= 0 && _currentIndex < experiments.Count)
                ? experiments[_currentIndex]
                : null;

        private void Start()
        {
            if (startOnPlay)
                StartSequence();
        }

        /// <summary>
        /// Begin running the experiment sequence from the first experiment.
        /// </summary>
        public void StartSequence()
        {
            if (_isRunning)
            {
                Debug.LogWarning("[ExperimentRunner] Sequence already running.");
                return;
            }

            if (experiments.Count == 0)
            {
                Debug.LogError("[ExperimentRunner] No experiments assigned.");
                return;
            }

            StartCoroutine(RunAllExperiments());
        }

        /// <summary>
        /// Skip the currently running experiment and move to the next one.
        /// Useful for debugging or a skip button.
        /// </summary>
        public void SkipCurrent()
        {
            if (CurrentExperiment != null)
            {
                Debug.Log($"[ExperimentRunner] Skipping: {CurrentExperiment.experimentName}");
                CurrentExperiment.OnEnd();
            }
        }

        private IEnumerator RunAllExperiments()
        {
            _isRunning = true;
            onSequenceStart?.Invoke();
            Debug.Log("[ExperimentRunner] Sequence started.");

            for (_currentIndex = 0; _currentIndex < experiments.Count; _currentIndex++)
            {
                Experiment exp = experiments[_currentIndex];
                if (exp == null)
                {
                    Debug.LogWarning($"[ExperimentRunner] Experiment at index {_currentIndex} is null, skipping.");
                    continue;
                }

                // Disable all other experiments' GameObjects so they don't interfere
                SetOnlyActive(exp);
                exp.em = this;

                exp.OnBegin();
                onExperimentStarted?.Invoke(exp);

                yield return StartCoroutine(exp.Run());

                exp.OnEnd();
                onExperimentFinished?.Invoke(exp);

                if (gapBetweenExperiments > 0f)
                    yield return new WaitForSeconds(gapBetweenExperiments);
            }

            _currentIndex = -1;
            _isRunning = false;
            onSequenceComplete?.Invoke();
            Debug.Log("[ExperimentRunner] Sequence complete.");
        }

        /// <summary>
        /// Enables the given experiment's GameObject and disables all others.
        /// </summary>
        private void SetOnlyActive(Experiment active)
        {
            foreach (var exp in experiments)
            {
                if (exp == null) continue;
                exp.gameObject.SetActive(exp == active);
            }
        }

        public void Log(ExperimentLogger.LogEntry entry)
        {
            
        }
    }
}