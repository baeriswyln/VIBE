using System;
using System.Collections;
using System.Collections.Generic;
using Resources.Scripts.Logging;
using UnityEngine;

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
    public class DataGatheringExperiment : StimuliExperiment
    {
        // --- State ---
        private Interactable _currentTarget;
        private Interactable _selectedStimulus;
        // -------------------------------------------------------------------

        public override void OnBegin()
        {
            base.OnBegin();
            _results.Clear();

            PinchHandler.Instance.onStimulusClicked.AddListener(OnStimulusClicked);

            if (feedbackText != null) feedbackText.text = "";
        }

        public override void OnEnd()
        {
            PinchHandler.Instance.onStimulusClicked.RemoveListener(OnStimulusClicked);

            // Clean up any remaining stimuli
            if (spawner)
                spawner.DestroyStimuli();

            // Clear UI
            if (trialInfoText) trialInfoText.text = "";
            if (feedbackText) feedbackText.text = "";

            base.OnEnd();
        }

        public override IEnumerator Run()
        {
            if (!spawner)
            {
                ExperimentLogger.Instance.Error(new ExperimentLog(GetType().Name, "No spawner assigned."));
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

            ExperimentLogger.Instance.Log(new ExperimentLog(GetType().Name, $"Finished: {_results.Count} events."));

            MarkComplete();
        }

        private IEnumerator RunSingleTrial(int trialNumber)
        {
            // --- Setup ---
            if (trialInfoText)
                trialInfoText.text = $"Trial {trialNumber} / {numberOfTrials}";

            if (feedbackText) feedbackText.text = "";

            // Pick a random stimulus as the target
            var spawned = spawner.spawnedStimuli;
            if (spawned == null || spawned.Count != stimuliPerTrial)
            {
                ExperimentLogger.Instance.Error(new ExperimentLog(GetType().Name,
                    $"Not enough Stimuli spawned for trial {trialNumber}."));
                yield break;
            }

            _currentTarget = spawned[0];
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
                TargetPosition = _currentTarget.transform.position,
                ResponseTime = responseTime,
            };
            _results.Add(result);
            onTrialComplete?.Invoke(result);

            ExperimentLogger.Instance.Log(new ExperimentLog(GetType().Name,
                $"trial={trialNumber} target={result.TargetId}  rt={result.ResponseTime:F3}s"));

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