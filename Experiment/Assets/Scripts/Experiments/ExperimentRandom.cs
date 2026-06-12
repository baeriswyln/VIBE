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
                ExperimentLogger.Instance.Error(new ExperimentLog(experimentName, "No spawner assigned."));
                yield break;
            }
            
            yield return PlayTextPages(textStart);
            if(trialInfoText) trialInfoText.gameObject.SetActive(false);
            if(endScheduled) yield break;

            while (true)
            {
                spawner.SpawnStimuli(stimuliPerTrial);
                var spawned = spawner.spawnedStimuli;
                if (spawned != null && spawned.Count == stimuliPerTrial)
                {
                    break;
                }
                ExperimentLogger.Instance.Error(new ExperimentLog(experimentName, $"Not enough Stimuli spawned."));
                spawner.DestroyStimuli();
            }
            PinchHandler.Instance.Active = true;
            for (int trial = 0; trial < numberOfTrials; trial++)
            {
                yield return StartCoroutine(RunSingleTrial(trial));
                if(endScheduled)
                {
                    spawner.DestroyStimuli();
                    yield break;
                }
                yield return WaitForNextStimulus();
            }
            PinchHandler.Instance.Active = false;
            
            spawner.DestroyStimuli();
            yield return PlayTextPages(textEnd);
            ExperimentLogger.Instance.Log(new ExperimentLog(experimentName, $"Finished: {_results.Count} events."));
            MarkComplete();
        }
        
    }
}