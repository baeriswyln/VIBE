using UnityEngine;
using UnityEngine.InputSystem.XR;
using UnityEngine.Rendering.Universal;

/// <summary>
/// Lets the experiment run without VR hardware: when no XR headset is active,
/// this disables the head-tracked pose driver on the gaze camera and swaps in
/// MouseLookController so the mouse steers the gaze raycast. It also turns off
/// GazeRaycaster.useEyeTracking and disables ViveEyeLslStreamer, since without
/// a headset the Vive eye-tracker API has nothing to talk to — leaving it on
/// makes both components poll/probe it every frame, which throws and logs
/// errors continuously and makes the game laggy.
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

        gazeRaycaster.useEyeTracking = false;

        var eyeLslStreamer = Object.FindFirstObjectByType<ViveEyeLslStreamer>();
        if (eyeLslStreamer != null)
            eyeLslStreamer.enabled = false;

        GameObject cam = gazeRaycaster.gameObject;

        var trackedPoseDriver = cam.GetComponent<TrackedPoseDriver>();
        if (trackedPoseDriver != null)
            trackedPoseDriver.enabled = false;

        // Disable stereo view
        var camera = cam.GetComponent<Camera>();
        if (camera != null)
            camera.stereoTargetEye = StereoTargetEyeMask.None;

        // Disable XR render pass
        var cameraData = cam.GetComponent<UniversalAdditionalCameraData>();
        if (cameraData != null)
            cameraData.allowXRRendering = false;

        if (cam.GetComponent<MouseLookController>() == null)
            cam.AddComponent<MouseLookController>();

        Debug.Log("[NonVRTestModeBootstrap] No XR headset detected — mouse-look test mode enabled " +
                  "(mouse steers gaze, left click selects).");
    }
}