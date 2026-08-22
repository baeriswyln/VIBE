using UnityEngine;
using UnityEngine.InputSystem.XR;

/// <summary>
/// Lets the experiment run without VR hardware: when no XR headset is active,
/// this disables the head-tracked pose driver on the gaze camera and swaps in
/// MouseLookController so the mouse steers the gaze raycast (GazeRaycaster
/// already falls back to plain camera-forward when eye tracking is unavailable).
/// The existing "ExpPinch" input action already fires select on a left mouse
/// click (see InputSystem_Actions.inputactions), so no change is needed there.
/// Requires no scene setup — runs automatically on scene load.
/// </summary>
public static class NonVRTestModeBootstrap
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Init()
    {
        if (UnityEngine.XR.XRSettings.isDeviceActive)
            return; // real headset connected and tracking — leave VR input untouched

        var gazeRaycaster = Object.FindFirstObjectByType<GazeRaycaster>();
        if (gazeRaycaster == null)
            return;

        GameObject cam = gazeRaycaster.gameObject;

        var trackedPoseDriver = cam.GetComponent<TrackedPoseDriver>();
        if (trackedPoseDriver != null)
            trackedPoseDriver.enabled = false;

        if (cam.GetComponent<MouseLookController>() == null)
            cam.AddComponent<MouseLookController>();

        Debug.Log("[NonVRTestModeBootstrap] No XR headset detected — mouse-look test mode enabled " +
                  "(mouse steers gaze, left click selects).");
    }
}