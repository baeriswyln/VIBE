using UnityEditor;
using UnityEngine;

namespace Editor
{
    public class SpectatorWindow : EditorWindow
    {
        public RenderTexture renderTexture;

        [MenuItem("Window/Spectator View")]
        public static void ShowWindow()
        {
            GetWindow<SpectatorWindow>("Spectator View");
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
            Repaint(); // force redraw every frame while playing
        }
    }
}