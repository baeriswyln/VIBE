using System;
using System.Collections;
using System.Collections.Generic;
using Leap;
using Logging;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using Random = UnityEngine.Random;

namespace Experiments
{
    public abstract class StimuliExperiment : Experiment
    {
        [Header("Events")] public UnityEvent<TrialResult> onTrialComplete;

        public PinchHandler.PinchSource currentPinchSource = PinchHandler.PinchSource.Leap;

        // --- State ---
        protected Interactable _currentTarget;
        protected Interactable _selectedStimulus;
        protected Interactable _gazedStimulus;
        protected float _gazedStimulusStart;
        
        // --- Stimuli order --
        protected List<bool> _errorList;
        protected List<bool> _observableList;

        public struct TrialResult
        {
            public int TrialNumber; // 1-indexed
            public DateTime Timestamp;
            public string TargetId;
            public float ResponseTime; // seconds from target visible to click
            public Vector3 TargetPosition;
        }


        protected readonly List<TrialResult> _results = new List<TrialResult>();

        /// <summary>Read-only access to the full trial history.</summary>
        public IReadOnlyList<TrialResult> Results => _results;

        protected StimuliExperiment(ExperimentConfig config) : base(config)
        {
            _errorList = new List<bool>();
            _observableList = new List<bool>();
            
            if (config != null)
            {
                for (var i = 0; i < config.numberOfStimuli; i++)
                {
                    _errorList.Add(i < config.numberOfErrors);
                    _observableList.Add(i < config.numberOfObserveStimuli);
                }

                _errorList.Shuffle();
                _observableList.Shuffle();
            }
        }

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
            if (ExperimentManager.Instance.spawner)
                ExperimentManager.Instance.spawner.DestroyStimuli();

            base.OnEnd();
        }

        protected IEnumerator WaitForNextStimulus()
        {
            if (ExperimentManager.Instance.interTrialIntervalMax == 0)
                yield break;

            if (ExperimentManager.Instance.interTrialIntervalMin == ExperimentManager.Instance.interTrialIntervalMax)
                yield return new WaitForSeconds(ExperimentManager.Instance.interTrialIntervalMin);
            else
            {
                var secondsToWait = Random.Range(ExperimentManager.Instance.interTrialIntervalMin, ExperimentManager.Instance.interTrialIntervalMax);
                yield return new WaitForSeconds(secondsToWait);
            }
        }

        protected IEnumerator RunSingleTrial(int trialNumber)
        {
            // Pick a random stimulus as the target
            var spawned = ExperimentManager.Instance.spawner.spawnedStimuli;
            if (spawned == null || spawned.Count != config.numberOfStimuli)
            {
                // todo update log
                ExperimentLogger.Instance.Error(new ExperimentLog("",
                    $"Not enough Stimuli spawned for trial {trialNumber}."));
                yield break;
            }

            if (config.isChoiceStimuli) _currentTarget = null;
            else
            {
                _currentTarget = spawned[Random.Range(0, spawned.Count)];

                _currentTarget.onGazeEnter += OnStimulusGazeEnter;
                _currentTarget.onGazeExit += OnStimulusGazeExit;
                _currentTarget.Highlight(_observableList[trialNumber]);
            }

            _selectedStimulus = null;
            _gazedStimulus = null;

            float responseStart = Time.time;
            
            // the first pinch on the target will be ignored (the logs will show it, but the player won't notice it)
            if (_errorList[trialNumber])
            {
                PinchHandler.Instance.hideNextPinch = true;
                while (!StimulusSuccessCondition())
                {
                    if (endScheduled)
                    {
                        PinchHandler.Instance.hideNextPinch = false;
                        yield break;
                    }
                    yield return null;
                }
            }

            // --- Wait for correct stimulus click or highlight ---
            while (!StimulusSuccessCondition())
            {
                if (endScheduled) yield break;
                yield return null;
            }

            // make the selected/gazed stimulus the current target - in this case, the player had to choose any stimulus
            if (!_currentTarget) _currentTarget = _selectedStimulus ? _selectedStimulus : _gazedStimulus;
            _currentTarget.onGazeEnter -= OnStimulusGazeEnter;
            _currentTarget.onGazeExit -= OnStimulusGazeExit;

            float responseTime = Time.time - responseStart;

            // --- Evaluate result ---
            var result = new TrialResult
            {
                TrialNumber = trialNumber,
                TargetId = _currentTarget.StimulusId,
                TargetPosition = _currentTarget.transform.position,
                ResponseTime = responseTime,
            };
            _results.Add(result);
            onTrialComplete?.Invoke(result);

            ExperimentLogger.Instance.Log(new ExperimentLog("",
                $"action={(_currentTarget.IsHighlighted ? "select" : "gazeSelect")},trial={trialNumber},target={result.TargetId},rt={result.ResponseTime:F3}s"));

            yield return new WaitForSeconds(ExperimentManager.Instance.feedbackDuration);

            // --- Cleanup ---
            ExperimentManager.Instance.spawner.ResetAll();
            _currentTarget = null;
        }

        private bool StimulusSuccessCondition()
        {
            return
                // for negative highlight, success condition is minimum gaze time
                (_gazedStimulus && _gazedStimulus.IsHighlightedNegative && 
                 Time.time - _gazedStimulusStart > ExperimentManager.Instance.gazeMinTime) ||
                // otherwise, wait for highlighted stimulus to be selected, or any stimulus select if the player must
                // make their own choice.
                (_selectedStimulus && (_selectedStimulus.IsHighlighted || config.isChoiceStimuli));
        }

        private void OnStimulusGazeEnter(Interactable gazed)
        {
            _gazedStimulus = gazed;
            _gazedStimulusStart = Time.time;
        }

        private void OnStimulusGazeExit(Interactable gazed)
        {
            _gazedStimulus = null;
        }

        private void OnStimulusClicked(Interactable clicked)
        {
            _selectedStimulus = clicked;
        }
    }
}