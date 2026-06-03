using System;
using System.Collections;
using System.Collections.Generic;
using Logging;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using Random = UnityEngine.Random;

namespace Experiments
{
    public abstract class StimuliExperiment : Experiment
    {
        [Header("Trial Configuration")]
        [Tooltip("How many stimuli to spawn per trial.")]
        public int stimuliPerTrial = 1;

        [Tooltip("Number of trials to run.")]
        public int numberOfTrials = 20;

        [Tooltip("Seconds to wait between trials.")]
        public float interTrialIntervalMin = .5f;
        public float interTrialIntervalMax = 2.5f;

        [Tooltip("Seconds to show feedback (correct/incorrect) before moving on.")]
        public float feedbackDuration = 1.0f;
        
        [Header("References")]
        [Tooltip("StimulusSpawner that handles placing the stimuli.")]
        public InteractableManager spawner;

        [Header("Events")]
        public UnityEvent<TrialResult> onTrialComplete;
        
        public PinchHandler.PinchSource currentPinchSource = PinchHandler.PinchSource.Leap;
        
        // --- State ---
        protected Interactable _currentTarget;
        protected Interactable _selectedStimulus;
        
        public struct TrialResult
        {
            public int TrialNumber;          // 1-indexed
            public DateTime Timestamp;
            public string TargetId;
            public float ResponseTime;       // seconds from target visible to click
            public Vector3 TargetPosition;
        }
        
        
        protected readonly List<TrialResult> _results = new List<TrialResult>();
        
        /// <summary>Read-only access to the full trial history.</summary>
        public IReadOnlyList<TrialResult> Results => _results;

        public override void OnBegin()
        {
            base.OnBegin();
            _results.Clear();
            PinchHandler.Instance.currentPinchSource = currentPinchSource;
            PinchHandler.Instance.onStimulusClicked.AddListener(OnStimulusClicked);
        }

        public override void OnEnd()
        {
            PinchHandler.Instance.onStimulusClicked.RemoveListener(OnStimulusClicked);

            // Clean up any remaining stimuli
            if (spawner)
                spawner.DestroyStimuli();
            
            base.OnEnd();
        }

        protected IEnumerator WaitForNextStimulus()
        {
            if (interTrialIntervalMax == 0)
                yield break;
            
            if (interTrialIntervalMin == interTrialIntervalMax)
                yield return new WaitForSeconds(interTrialIntervalMin);
            else
            {
                var secondsToWait = Random.Range(interTrialIntervalMin, interTrialIntervalMax);
                yield return new WaitForSeconds(secondsToWait);
            }
        }
        
        protected IEnumerator RunSingleTrial(int trialNumber)
        {
            // Pick a random stimulus as the target
            var spawned = spawner.spawnedStimuli;
            if (spawned == null || spawned.Count != stimuliPerTrial)
            {
                ExperimentLogger.Instance.Error(new ExperimentLog(GetType().Name, $"Not enough Stimuli spawned for trial {trialNumber}."));
                yield break;
            }

            _currentTarget = spawned[Random.Range(0, spawned.Count)];
            _currentTarget.Highlight();

            _selectedStimulus = null;

            float responseStart = Time.time;

            // --- Wait for correct stimulus click ---
            while (!(_selectedStimulus && _selectedStimulus == _currentTarget))
            {
                if (endScheduled) yield break;
                yield return null;
            }

            float responseTime = Time.time - responseStart;

            // --- Evaluate result ---
            var result = new TrialResult
            {
                TrialNumber = trialNumber,
                TargetId = _currentTarget.StimulusId,
                TargetPosition =  _currentTarget.transform.position,
                ResponseTime = responseTime,
            };
            _results.Add(result);
            onTrialComplete?.Invoke(result);

            ExperimentLogger.Instance.Log(new ExperimentLog(GetType().Name, $"trial={trialNumber} target={result.TargetId}  rt={result.ResponseTime:F3}s"));
            
            yield return new WaitForSeconds(feedbackDuration);

            // --- Cleanup ---
            spawner.ResetAll();
            _currentTarget = null;
        }
        
        private void OnStimulusClicked(Interactable clicked)
        {
            _selectedStimulus = clicked;
        }
    }
}