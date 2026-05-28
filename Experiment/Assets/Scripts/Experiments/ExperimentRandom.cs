using System.Collections;
using Logging;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Experiments
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
    public class ExperimentRandom : StimuliExperiment
    {


        public override IEnumerator Run()
        {
            if (!spawner)
            {
                ExperimentLogger.Instance.Error(new ExperimentLog(GetType().Name, "No spawner assigned."));
                yield break;
            }

            while (true)
            {
                spawner.SpawnStimuli(stimuliPerTrial);
                var spawned = spawner.spawnedStimuli;
                if (spawned == null || spawned.Count != stimuliPerTrial)
                {
                    ExperimentLogger.Instance.Error(new ExperimentLog(GetType().Name, $"Not enough Stimuli spawned."));
                    break;
                }
                spawner.DestroyStimuli();
            }
            for (int trial = 1; trial <= numberOfTrials; trial++)
            {
                yield return StartCoroutine(RunSingleTrial(trial));
                yield return WaitForNextStimulus();
            }
            spawner.DestroyStimuli();

            ExperimentLogger.Instance.Log(new ExperimentLog(GetType().Name, $"Finished: {_results.Count} events."));

            MarkComplete();
        }

        private IEnumerator RunSingleTrial(int trialNumber)
        {
            // --- Setup ---
            if (trialInfoText) trialInfoText.text = "";

            // Pick a random stimulus as the target
            var spawned = spawner.spawnedStimuli;
            _currentTarget = spawned[Random.Range(0, spawned.Count)];
            _currentTarget.Highlight();

            _selectedStimulus = null;

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