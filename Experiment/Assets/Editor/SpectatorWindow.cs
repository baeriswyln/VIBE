using UnityEditor;
using UnityEngine;

namespace Editor
{
    public class SpectatorWindow : EditorWindow
    {
        private const double RepaintIntervalSeconds = 1.0 / 15.0;

        public RenderTexture renderTexture;
        private double lastRepaintTime;
        private Camera sourceCamera;
        private RenderTexture trackedTexture;

        [MenuItem("Window/Spectator View")]
        public static void ShowWindow()
        {
            GetWindow<SpectatorWindow>("Spectator View");
        }

        void OnEnable()
        {
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
            SetSourceCameraEnabled(EditorApplication.isPlaying);
        }

        void OnDisable()
        {
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            SetSourceCameraEnabled(false); // window closed (or reloading) — stop paying for a render nobody sees
        }

        private void OnPlayModeStateChanged(PlayModeStateChange change)
        {
            if (change == PlayModeStateChange.EnteredPlayMode)
                SetSourceCameraEnabled(true);
        }

        void OnGUI()
        {
            renderTexture = (RenderTexture)EditorGUILayout.ObjectField(
                "Render Texture", renderTexture, typeof(RenderTexture), false);

            if (renderTexture != null)
            {
                Rect rect = GUILayoutUtility.GetRect(position.width, position.height - 40);
                EditorGUI.DrawPreviewTexture(rect, renderTexture);
            }
        }

        void Update()
        {
            if (!EditorApplication.isPlaying || renderTexture == null)
                return;

            SetSourceCameraEnabled(true);

            // reduce frame rate
            double now = EditorApplication.timeSinceStartup;
            if (now - lastRepaintTime < RepaintIntervalSeconds)
                return;

            lastRepaintTime = now;
            Repaint();
        }
        
        private void SetSourceCameraEnabled(bool value)
        {
            if (renderTexture != trackedTexture)
            {
                sourceCamera = null;
                trackedTexture = renderTexture;
            }

            if (renderTexture == null)
                return;

            if (sourceCamera == null)
            {
                foreach (var cam in Resources.FindObjectsOfTypeAll<Camera>())
                {
                    if (cam.targetTexture == renderTexture && cam.gameObject.scene.IsValid())
                    {
                        sourceCamera = cam;
                        break;
                    }
                }
            }

            if (sourceCamera != null)
                sourceCamera.enabled = value;
        }
    }
}