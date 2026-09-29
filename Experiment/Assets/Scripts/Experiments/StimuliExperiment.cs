using System;
using System.Collections;
using System.Collections.Generic;
using Leap;
using Logging;
using Newtonsoft.Json;
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
        protected Interactable _pinchedStimulus;
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

        public override void SetConfig(ExperimentConfig cfg)
        {
            config = cfg;
            _errorList = new List<bool>();
            _observableList = new List<bool>();
            
            if (config != null)
            {
                for (var i = 0; i < config.numberOfTrials; i++)
                {
                    _errorList.Add(i < config.numberOfErrors);
                    _observableList.Add(i < config.numberOfObserveTrials);
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
            if (spawner)
                spawner.DestroyStimuli();

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
            var spawned = spawner.spawnedStimuli;
            if (spawned == null || spawned.Count != config.numberOfStimuli)
            {
                // todo update log
                ExperimentLogger.Instance.Error(new ExperimentLog("",
                    $"Not enough Stimuli spawned for trial {trialNumber}."));
                yield break;
            }

            // todo: test if choice and gaze is compatible.
            if (config.isChoiceStimuli)
            {
                _currentTarget = null;
                foreach (var s in spawned)
                {
                    s.onGazeEnter += OnStimulusGazeEnter;
                    s.onGazeExit += OnStimulusGazeExit;
                }
            }
            else
            {
                _currentTarget = spawned[Random.Range(0, spawned.Count)];

                _currentTarget.onGazeEnter += OnStimulusGazeEnter;
                _currentTarget.onGazeExit += OnStimulusGazeExit;
                _currentTarget.Highlight(_observableList[trialNumber]);
            }

            _pinchedStimulus = null;
            _gazedStimulus = null;

            float responseStart = Time.time;
            // safety reset
            PinchHandler.Instance.hideNextPinch = false;
            
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
                // reset state to prepare for real pinch
                _pinchedStimulus = null;
                _gazedStimulus = null;
                PinchHandler.Instance.hideNextPinch = false;
            }

            // --- Wait for correct stimulus click or highlight ---
            Interactable selectedStimulus = null;
            while (!selectedStimulus)
            {
                if (endScheduled) yield break;
                selectedStimulus = StimulusSuccessCondition();
                yield return null;
            }

            foreach (var s in spawned)
            {
                s.onGazeEnter -= OnStimulusGazeEnter;
                s.onGazeExit -= OnStimulusGazeExit;   
            }

            float responseTime = Time.time - responseStart;

            // --- Evaluate result ---
            var result = new TrialResult
            {
                TrialNumber = trialNumber,
                TargetId = selectedStimulus.StimulusId,
                TargetPosition = selectedStimulus.transform.position,
                ResponseTime = responseTime,
            };
            _results.Add(result);
            onTrialComplete?.Invoke(result);

            ExperimentLogger.Instance.Log(new ExperimentLog(ToString(),
                JsonConvert.SerializeObject(new
                    {
                        Action = selectedStimulus.IsHighlighted ? "select" : "gazeSelect",
                        Trial = trialNumber,
                        Target = result.TargetId,
                        ResponseTime = result.ResponseTime,
                        ExperimentConfig = ExperimentManager.Instance.CurrentExperiment?.config,
                    })));
            if (selectedStimulus.IsHighlightedNegative)
            {
                // special pinch log for gaze select
                ExperimentLogger.Instance.Log(new PinchLog(selectedStimulus, PinchLog.PinchType.GAZE));
            }

            if (ExperimentManager.Instance.CurrentExperiment?.config.feedbackIsDelayed ?? false)
                yield return new WaitForSeconds(PinchHandler.Instance.feedbackDelay);
            yield return new WaitForSeconds(ExperimentManager.Instance.feedbackDuration);

            // --- Cleanup ---
            spawner.ResetAll();
            _currentTarget = null;
        }

        private Interactable StimulusSuccessCondition()
        {
            // for negative highlight, success condition is minimum gaze time
            if (_gazedStimulus && _gazedStimulus.IsHighlightedNegative &&
                Time.time - _gazedStimulusStart > ExperimentManager.Instance.gazeMinTime)
                return _gazedStimulus;
            
            // otherwise, wait for highlighted stimulus to be selected, or any stimulus select if the player must
            // make their own choice.
            if (_pinchedStimulus && (_pinchedStimulus.IsHighlighted || config.isChoiceStimuli))
                return _pinchedStimulus;

            return null;
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
            _pinchedStimulus = clicked;
        }
    }
}