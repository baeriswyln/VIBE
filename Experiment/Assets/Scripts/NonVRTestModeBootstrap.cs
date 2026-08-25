using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem.XR;
using UnityEngine.Rendering.Universal;
using UnityEngine.XR.Management;

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
    // XRSettings.isDeviceActive can still read false for the first few frames
    // after a real headset's loader has initialized, because the runtime
    // (e.g. OpenXR) finishes its session handshake asynchronously. Give it a
    // window to come up before concluding no headset is present.
    private const float DeviceActivationTimeoutSeconds = 10f;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Init()
    {
        var xrManager = XRGeneralSettings.Instance != null ? XRGeneralSettings.Instance.Manager : null;
        if (xrManager == null || xrManager.activeLoader == null)
        {
            // No XR loader ever came up, so no runtime/headset was found at
            // all — this is known immediately, nothing to wait for.
            EnableMouseLookMode();
            return;
        }

        if (UnityEngine.XR.XRSettings.isDeviceActive)
            return; // real headset connected and already tracking — leave VR input untouched

        var waiter = new GameObject("[NonVRTestModeBootstrap] XR Activation Waiter").AddComponent<XrActivationWaiter>();
        Object.DontDestroyOnLoad(waiter.gameObject);
        waiter.Begin(DeviceActivationTimeoutSeconds);
    }

    private sealed class XrActivationWaiter : MonoBehaviour
    {
        public void Begin(float timeoutSeconds) => StartCoroutine(WaitRoutine(timeoutSeconds));

        private IEnumerator WaitRoutine(float timeoutSeconds)
        {
            float deadline = Time.unscaledTime + timeoutSeconds;
            while (Time.unscaledTime < deadline)
            {
                if (UnityEngine.XR.XRSettings.isDeviceActive)
                {
                    Destroy(gameObject);
                    yield break;
                }
                yield return null;
            }

            Debug.LogWarning("[NonVRTestModeBootstrap] XR loader initialized but headset never became active " +
                              $"within {timeoutSeconds:0}s — falling back to mouse-look test mode.");
            EnableMouseLookMode();
            Destroy(gameObject);
        }
    }

    private static void EnableMouseLookMode()
    {
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
