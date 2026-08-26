using System.Collections;
using System.Collections.Generic;
using Leap;
using Logging;
using UnityEngine;

namespace Experiments
{
    public class ExperimentStatic : StimuliExperiment
    {
        [Header("Stimulus Spawn Configuration")]
        public Vector3 spawnCenter = new(0, 1, 5);
        public float spawnRadius = 0.5f;
        
        private bool _respawnBetweenRounds = true;

        public override IEnumerator Run(bool  skipEndPause = false)
        {
            _respawnBetweenRounds = config.numberOfStimuli == 1;
            
            if (!spawner)
            {
                ExperimentLogger.Instance.Error(new ExperimentLog(ToString(), "No spawner assigned."));
                yield break;
            }

            yield return ExperimentManager.Instance.PlayExperimentInstructions(config);
            if (endScheduled) yield break;
            
            spawner.SpawnStimulusInCircle(config, spawnCenter, spawnRadius);

            PinchHandler.Instance.Active = true;
            for (var trial = 0; trial < config.numberOfTrials; trial++)
            {
                if (config.isChoiceStimuli) spawner.RandomizeColors();
                yield return RunSingleTrial(trial);
                if(endScheduled)
                {
                    spawner.DestroyStimuli();
                    yield break;
                }
                
                // shorter wait after last stimulus
                if (trial < config.numberOfTrials - 1)
                {
                    if (_respawnBetweenRounds) spawner.DestroyStimuli();
                    yield return WaitForNextStimulus();
                    if (_respawnBetweenRounds) spawner.SpawnStimulusInCircle(config, spawnCenter, spawnRadius);
                }
                else yield return new WaitForSeconds(1);
            }
            PinchHandler.Instance.Active = false;
            
            spawner.DestroyStimuli();
            if (skipEndPause)
            {
                yield return ExperimentManager.Instance.PlayTextPages(new[] { "End of the round, short break." });
            }
            ExperimentLogger.Instance.Log(new ExperimentLog(ToString(), $"Finished: {_results.Count} events."));
            MarkComplete();
        }
    }
}