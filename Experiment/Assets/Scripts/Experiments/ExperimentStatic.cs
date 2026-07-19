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
        public float spawnRadius;
        
        private bool _respawnBetweenRounds = true;

        public ExperimentStatic(ExperimentConfig config) : base(config)
        {
        }

        public override IEnumerator Run()
        {
            _respawnBetweenRounds = config.numberOfStimuli == 1;
            
            var spawner = ExperimentManager.Instance.spawner; 
            if (!spawner)
            {
                ExperimentLogger.Instance.Error(new ExperimentLog(ToString(), "No spawner assigned."));
                yield break;
            }

            yield return ExperimentManager.Instance.PlayExperimentInstructions(config);
            if (endScheduled) yield break;
            
            spawner.SpawnStimulusInCircle(config, spawnCenter, spawnRadius);
            if (config.isChoiceStimuli) spawner.RandomizeColors();

            PinchHandler.Instance.Active = true;
            for (var i = 0; i < config.numberOfTrials; i++)
            {
                yield return RunSingleTrial(i);
                if(endScheduled)
                {
                    spawner.DestroyStimuli();
                    yield break;
                }
                if(_respawnBetweenRounds) spawner.DestroyStimuli();
                yield return WaitForNextStimulus();
                if(_respawnBetweenRounds) spawner.SpawnStimulusInCircle(config, spawnCenter, spawnRadius);
                if (config.isChoiceStimuli) spawner.RandomizeColors();
            }
            PinchHandler.Instance.Active = false;
            
            spawner.DestroyStimuli();
            // yield return PlayTextPages(textEnd);
            ExperimentLogger.Instance.Log(new ExperimentLog(ToString(), $"Finished: {_results.Count} events."));
            MarkComplete();
        }
    }
}