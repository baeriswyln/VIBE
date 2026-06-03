using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Logging;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;

namespace Experiments
{
    public class ExperimentManager : MonoBehaviour
    {
        public static ExperimentManager Instance { get; private set; }
        
        [Header("Experiment Sequence")] [Tooltip("Experiments will run in this order.")]
        public List<Experiment> experiments = new List<Experiment>();

        [Header("Start Behaviour")]
        [Tooltip("Start the sequence automatically on Play. If false, call StartSequence() manually.")]
        public bool startOnPlay = true;

        [Tooltip("Seconds to wait between experiments.")]
        public float gapBetweenExperiments = 1f;
        
        public InputActionAsset inputActionAsset;
        
        [SerializeField] private GazeRaycaster gazeRaycaster;
        [SerializeField] private GameObject pauseUi;

        [Header("Events")] public UnityEvent onSequenceStart;
        public UnityEvent onSequenceComplete;
        public UnityEvent<Experiment> onExperimentStarted;
        public UnityEvent<Experiment> onExperimentFinished;

        // State
        private int _currentIndex = -1;
        private bool _isRunning;
        private bool _isPaused = false;

        DateTime sessionStartUtc = DateTime.UtcNow;
        Stopwatch stopwatch = Stopwatch.StartNew();

        public bool IsRunning => _isRunning;
        public int CurrentIndex => _currentIndex;

        public Experiment CurrentExperiment =>
            (_currentIndex >= 0 && _currentIndex < experiments.Count)
                ? experiments[_currentIndex]
                : null;

        // Singleton initialization
        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        
        private void Start()
        {
            if (startOnPlay)
                StartSequence();
        }

        private void OnEnable()
        {
            inputActionAsset.FindActionMap("Experiment").Enable();
            InputSystem.actions.FindAction("ExpNext").performed += SkipCurrent;
            InputSystem.actions.FindAction("ExpPrevious").performed += GoToPrevious;
            InputSystem.actions.FindAction("ExpRestart").performed += RestartCurrent;
            InputSystem.actions.FindAction("ExpPause").performed += TogglePause;
        }

        private void OnDisable()
        {
            InputSystem.actions.FindAction("ExpPause").performed -= TogglePause;
            InputSystem.actions.FindAction("ExpRestart").performed -= RestartCurrent;
            InputSystem.actions.FindAction("ExpPrevious").performed -= GoToPrevious;
            InputSystem.actions.FindAction("ExpNext").performed -= SkipCurrent;
            inputActionAsset.FindActionMap("Experiment").Disable();
        }

        /// <summary>
        /// Begin running the experiment sequence from the first experiment.
        /// </summary>
        public void StartSequence()
        {
            if (_isRunning)
            {
                ExperimentLogger.Instance.Warn(new ExperimentManagerLog("SEQUENCE_ALREADY_RUNNING", ""));
                return;
            }

            if (experiments.Count == 0)
            {
                ExperimentLogger.Instance.Error(new ExperimentManagerLog("NO_EXPERIMENT", "No experiments assigned."));
                return;
            }

            StartCoroutine(RunAllExperiments());
        }

        /// <summary>
        /// Skip the currently running experiment and move to the next one.
        /// Useful for debugging or a skip button.
        /// </summary>
        public void SkipCurrent(InputAction.CallbackContext context)
        {
            if (CurrentExperiment != null)
            {
                ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_SKIP", CurrentExperiment.experimentName));
                CurrentExperiment.endScheduled = true;
            }
        }

        public void RestartCurrent(InputAction.CallbackContext context)
        {
            if (CurrentExperiment != null)
            {
                ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_RESTART", CurrentExperiment.experimentName));
                CurrentExperiment.endScheduled = true;
                _currentIndex -= 1;
            }
        }
        
        public void GoToPrevious(InputAction.CallbackContext context)
        {
            if (CurrentExperiment != null)
            {
                ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_PREVIOUS", CurrentExperiment.experimentName));
                CurrentExperiment.endScheduled = true;
                _currentIndex = Math.Max(-1, _currentIndex-2);
            }
        }

        public void TogglePause(InputAction.CallbackContext context)
        {
            if (CurrentExperiment != null)
            {
                ExperimentLogger.Instance.Log(new ExperimentManagerLog("TOGGLE_PAUSE", CurrentExperiment.experimentName));

                if (_isPaused)
                {
                    Time.timeScale = 1;
                    AudioListener.pause = false;
                    gazeRaycaster.Activate();
                    pauseUi.SetActive(false);
                }
                else
                {
                    Time.timeScale = 0;
                    AudioListener.pause = true;
                    gazeRaycaster.Deactivate();
                    pauseUi.SetActive(true);
                }
                _isPaused = !_isPaused;
            }
        }

        private IEnumerator RunAllExperiments()
        {
            _isRunning = true;
            onSequenceStart?.Invoke();
            ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_START", ""));

            for (_currentIndex = 0; _currentIndex < experiments.Count; _currentIndex++)
            {
                Experiment exp = experiments[_currentIndex];
                if (!exp)
                {
                    ExperimentLogger.Instance.Warn(new ExperimentManagerLog("MISSING_EXPERIMENT", $"Experiment at index {_currentIndex} is null, skipping."));
                    continue;
                }

                // Disable all other experiments' GameObjects so they don't interfere
                SetOnlyActive(exp);

                exp.OnBegin();
                onExperimentStarted?.Invoke(exp);

                yield return StartCoroutine(exp.Run());

                exp.OnEnd();
                onExperimentFinished?.Invoke(exp);

                if (gapBetweenExperiments > 0f && !exp.endScheduled && !exp.skipEndOfExperimentPause)
                    yield return new WaitForSeconds(gapBetweenExperiments);
            }

            _currentIndex = -1;
            _isRunning = false;
            onSequenceComplete?.Invoke();
            ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_COMPLETE", ""));
        }

        /// <summary>
        /// Enables the given experiment's GameObject and disables all others.
        /// </summary>
        private void SetOnlyActive(Experiment active)
        {
            foreach (var exp in experiments)
            {
                if (exp == null) continue;
                exp.gameObject.SetActive(exp == active);
            }
        }

        protected class ExperimentManagerLog : EventLog
        {
            private string _message;
            
            public ExperimentManagerLog(string name, string message)
            {
                eventName = name;
                _message = message;
            }

            protected override string GetPayload()
            {
                return JsonConvert.SerializeObject(new
                {
                    Message = _message,
                });
            }
        }
    }
}