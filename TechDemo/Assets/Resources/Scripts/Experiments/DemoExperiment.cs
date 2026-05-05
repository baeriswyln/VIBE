using System;
using System.Collections;
using System.Collections.Generic;
using Resources.Scripts.Logging;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using Random = UnityEngine.Random;

namespace Resources.Scripts.Experiments
{
    /// <summary>
    /// Spawns N stimuli, highlights one as the target, and waits for the user to
    /// select (pinch) the correct one. Repeats for a configurable number of trials.
    ///
    /// Each trial:
    ///   1. Spawn stimuli inside the bounding box
    ///   2. Randomly pick one to be the target; highlight it distinctively
    ///   3. Wait for the user to select a stimulus (any stimulus)
    ///   4. Record whether the correct target was selected
    ///   5. Brief feedback + pause
    ///   6. Clear and start the next trial
    /// </summary>
    public class DemoExperiment : Experiment
    {
        [Header("Trial Configuration")]
        [Tooltip("How many stimuli to spawn per trial.")]
        public int stimuliPerTrial = 10;

        [Tooltip("Number of trials to run.")]
        public int numberOfTrials = 20;

        [Tooltip("Seconds to wait between trials.")]
        public float interTrialInterval = 1.5f;

        [Tooltip("Seconds to show feedback (correct/incorrect) before moving on.")]
        public float feedbackDuration = 1.0f;

        [Tooltip("Seconds to show target highlight before the user can click (helps them notice the target).")]
        public float previewDuration = 0.5f;

        [Tooltip("Max seconds to wait for a response before marking trial as timeout.")]
        public float trialTimeout = 10f;

        [Header("References")]
        [Tooltip("StimulusSpawner that handles placing the stimuli.")]
        public InteractableManager spawner;

        [Tooltip("PinchDetector that fires onStimulusClicked.")]
        public PinchDetector pinchDetector;

        [Header("Target Highlighting")]
        [Tooltip("Color used to mark the target stimulus (different from gaze highlight).")]
        public Color targetColor = new Color(1f, 0.2f, 0.2f, 1f);

        [Tooltip("Scale multiplier for the target (on top of base scale).")]
        public float targetScaleMultiplier = 1.3f;

        [Header("Feedback UI (Optional)")]
        [Tooltip("World-space text for trial info (e.g. 'Trial 3 / 20').")]
        public TMP_Text trialInfoText;

        [Tooltip("World-space text for feedback ('Correct!' / 'Wrong').")]
        public TMP_Text feedbackText;

        [Header("Events")]
        public UnityEvent<TrialResult> onTrialComplete;

        // --- State ---
        private Interactable _currentTarget;
        private Interactable _selectedStimulus;
        private readonly List<TrialResult> _results = new List<TrialResult>();

        public struct TrialResult
        {
            public int TrialNumber;          // 1-indexed
            public DateTime Timestamp;
            public string TargetId;
            public float ResponseTime;       // seconds from target visible to click
            public Vector3 TargetPosition;
        }

        /// <summary>Read-only access to the full trial history.</summary>
        public IReadOnlyList<TrialResult> Results => _results;

        // -------------------------------------------------------------------

        public override void OnBegin()
        {
            base.OnBegin();
            _results.Clear();

            if (pinchDetector != null)
                pinchDetector.onStimulusClicked.AddListener(OnStimulusClicked);

            if (feedbackText != null) feedbackText.text = "";
        }

        public override void OnEnd()
        {
            if (pinchDetector != null)
                pinchDetector.onStimulusClicked.RemoveListener(OnStimulusClicked);

            // Clean up any remaining stimuli
            if (spawner != null)
                spawner.DestroyStimuli();

            // Clear UI
            if (trialInfoText != null) trialInfoText.text = "";
            if (feedbackText != null) feedbackText.text = "";

            base.OnEnd();
        }

        public override IEnumerator Run()
        {
            if (spawner == null)
            {
                em.LogError(new ExperimentLog(GetType().Name, "No spawner assigned."));
                yield break;
            }

            
            spawner.SpawnStimuli(stimuliPerTrial);
            for (int trial = 1; trial <= numberOfTrials; trial++)
            {
                yield return StartCoroutine(RunSingleTrial(trial));

                if (interTrialInterval > 0f)
                    yield return new WaitForSeconds(interTrialInterval);
            }
            spawner.DestroyStimuli();

            em.Log(new ExperimentLog(GetType().Name, $"Finished: {_results.Count} events."));

            MarkComplete();
        }

        private IEnumerator RunSingleTrial(int trialNumber)
        {
            // --- Setup ---
            if (trialInfoText != null)
                trialInfoText.text = $"Trial {trialNumber} / {numberOfTrials}";

            if (feedbackText != null) feedbackText.text = "";

            // Pick a random stimulus as the target
            var spawned = spawner.spawnedStimuli;
            if (spawned == null || spawned.Count != stimuliPerTrial)
            {
                em.LogError(new ExperimentLog(GetType().Name, $"Not enough Stimuli spawned for trial {trialNumber}."));
                yield break;
            }

            _currentTarget = spawned[Random.Range(0, spawned.Count)];
            _currentTarget.Highlight();

            _selectedStimulus = null;

            // Brief preview so the user notices the target
            yield return new WaitForSeconds(previewDuration);

            float responseStart = Time.time;

            // --- Wait for correct stimulus click ---
            while (!(_selectedStimulus && _selectedStimulus == _currentTarget)) 
            {
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

            em.Log(new ExperimentLog(GetType().Name, $"trial={trialNumber} target={result.TargetId}  rt={result.ResponseTime:F3}s"));
            
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