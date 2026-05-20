using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.Events;

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
        public float interTrialInterval = .5f;

        [Tooltip("Seconds to show feedback (correct/incorrect) before moving on.")]
        public float feedbackDuration = 1.0f;

        [Tooltip("Seconds to show target highlight before the user can click (helps them notice the target).")]
        public float previewDuration = 0.2f;
        
        [Header("References")]
        [Tooltip("StimulusSpawner that handles placing the stimuli.")]
        public InteractableManager spawner;
        
        [Header("Feedback UI (Optional)")]
        [Tooltip("World-space text for trial info (e.g. 'Trial 3 / 20').")]
        public TMP_Text trialInfoText;

        [Tooltip("World-space text for feedback ('Correct!' / 'Wrong').")]
        public TMP_Text feedbackText;

        [Header("Events")]
        public UnityEvent<TrialResult> onTrialComplete;
        
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
    }
}