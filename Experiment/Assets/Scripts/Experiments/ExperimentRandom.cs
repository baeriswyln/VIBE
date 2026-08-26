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
        public override IEnumerator Run(bool skipEndPause = false)
        {
            if (!spawner)
            {
                ExperimentLogger.Instance.Error(new ExperimentLog(ToString(), "No spawner assigned."));
                yield break;
            }

            yield return ExperimentManager.Instance.PlayExperimentInstructions(config);
            if(endScheduled) yield break;

            while (true)
            {
                spawner.SpawnStimuli(config);
                if (config.isChoiceStimuli) spawner.RandomizeColors();
                
                var spawned = spawner.spawnedStimuli;
                if (spawned != null && spawned.Count == config.numberOfStimuli)
                {
                    break;
                }
                ExperimentLogger.Instance.Error(new ExperimentLog(ToString(), $"Not enough Stimuli spawned."));
                spawner.DestroyStimuli();
            }
            
            PinchHandler.Instance.Active = true;
            for (int trial = 0; trial < config.numberOfTrials; trial++)
            {
                if (config.isChoiceStimuli) spawner.RandomizeColors();
                yield return RunSingleTrial(trial);
                if(endScheduled)
                {
                    spawner.DestroyStimuli();
                    yield break;
                }
                // shorter wait after last stimulus
                if (trial < config.numberOfTrials-1) yield return WaitForNextStimulus();
                else yield return new WaitForSeconds(1);
            }
            PinchHandler.Instance.Active = false;
            
            spawner.DestroyStimuli();
            if (!skipEndPause)
            {
                yield return ExperimentManager.Instance.PlayTextPages(new[] { "End of the round, short break." });
            }
            ExperimentLogger.Instance.Log(new ExperimentLog(ToString(), $"Finished: {_results.Count} events."));
            MarkComplete();
        }
        
    }
}