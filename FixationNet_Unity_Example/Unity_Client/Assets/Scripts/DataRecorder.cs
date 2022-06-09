using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class DataRecorder : MonoBehaviour
{
    public GameObject TaskObjects;
    public Camera headCamera;
    // the number of data item in one recording
    public int dataNumber = 40;
    // time rate to sample the data (hz)
    float sampleRate = 100;
    // time offset (ms) used to correct the sampling time, it depends on how fast your machine runs
    float timeOffset = 0;
    Queue<string> gazeData;
    Queue<string> headData;
    Queue<string> taskData;
    public string recordingsString;
    public bool Running;


    // Start is called before the first frame update
    void Start()
    {
        gazeData = new Queue<string>();
        headData = new Queue<string>();
        taskData = new Queue<string>();
        recordingsString = null;
        Running = true;
        StartCoroutine(RecordData());
    }


    IEnumerator RecordData()
    {
        WaitForSecondsRealtime waitTime = new WaitForSecondsRealtime(1f / sampleRate - timeOffset / 1000);

        while (Running)
        {
            SampleData();
            yield return waitTime;
        }
    }

    void SampleData()
    {
        // Timestamp
        System.TimeSpan timeSpan = System.DateTime.Now - new System.DateTime(1970, 1, 1, 0, 0, 0);
        long time = (long)timeSpan.TotalMilliseconds - 8 * 60 * 60 * 1000;
        string timeStamp = time.ToString();


        // Eye Gaze Data
        // In real applications, get the gaze data from your eye tracker
        // (0, 0) at Bottom-left, (1, 1) at Top-right
        float gazeX = Random.value;
        float gazeY = Random.value;
        Coordinate screenCoord;
        screenCoord.posX = gazeX;
        screenCoord.posY = gazeY;
        // FixationNet Model uses angular coordinates as inputs.
        Coordinate angularCoord = ScreenCoord2AngularCoord(screenCoord);
        string gazeInfo = angularCoord.posX.ToString("f2") + "," + angularCoord.posY.ToString("f2");


        // Head Rotation Velocity
        float headVelX = headCamera.GetComponent<CalculateHeadVelocity>().headVelX;
        float headVelY = headCamera.GetComponent<CalculateHeadVelocity>().headVelY;
        string headInfo = headVelX.ToString("f2") + "," + headVelY.ToString("f2");

        string taskInfo = TaskObjects.GetComponent<TrackObjects>().trackedObjectsString;
              
        if (gazeData.Count < dataNumber)
        {
            gazeData.Enqueue(gazeInfo);
            headData.Enqueue(headInfo);
            taskData.Enqueue(taskInfo);
            if (gazeData.Count == dataNumber)
            {
                // Collect data in time descending order (t, t-10, t-20, ...).
                recordingsString = null;
                foreach (string data in taskData)
                    recordingsString = data + "," + recordingsString;
                foreach (string data in headData)
                    recordingsString = data + "," + recordingsString;
                foreach (string data in gazeData)
                    recordingsString = data + "," + recordingsString;

                recordingsString = timeStamp + "," + recordingsString;
                //Debug.Log("recordingsString: " + recordingsString);
            }

        }
        else if (gazeData.Count == dataNumber)
        {
            // renew the info
            gazeData.Dequeue();
            headData.Dequeue();
            taskData.Dequeue();
            gazeData.Enqueue(gazeInfo);
            headData.Enqueue(headInfo);
            taskData.Enqueue(taskInfo);

            // Collect data in time descending order (t, t-10, t-20, ...).
            recordingsString = null;
            foreach (string data in taskData)
                recordingsString = data + "," + recordingsString;
            foreach (string data in headData)
                recordingsString = data + "," + recordingsString;
            foreach (string data in gazeData)
                recordingsString = data + "," + recordingsString;

            recordingsString = timeStamp + "," + recordingsString;
            //Debug.Log("recordingsString: " + recordingsString);
        }
    }

    struct Coordinate
    {
        public float posX;
        public float posY;
    }

    // Transform the screen coordinates to angular coordinates
    // Screen coordinates: (0, 0) at Bottom-left, (1, 1) at Top-right
    // Angular coordinates: (0 deg, 0 deg) at screen center
    Coordinate ScreenCoord2AngularCoord(Coordinate ScreenCoord)
    {
        // the parameters of our Hmd (HTC Vive).
        // the vertical fov of HTC VIVE PRE is fixed to 110 degree.       
        float VerticalFOV = Mathf.PI * 110 / 180;
        // the size of a half screen.
        float ScreenWidth = 1080;
        float ScreenHeight = 1200;

        // The Position of the Screen Center.
        float ScreenCenterX = 0.5f * ScreenWidth;
        float ScreenCenterY = 0.5f * ScreenHeight;

        // the distance between eye and the screen center.
        float ScreenDist = 0.5f * ScreenHeight / Mathf.Tan(VerticalFOV / 2);

        ScreenCoord.posX *= ScreenWidth;
        ScreenCoord.posY *= ScreenHeight;

        Coordinate AngularCoord;
        AngularCoord.posX = Mathf.Atan((ScreenCoord.posX - ScreenCenterX) / ScreenDist) / Mathf.PI * 180;
        AngularCoord.posY = Mathf.Atan((ScreenCoord.posY - ScreenCenterY) / ScreenDist) / Mathf.PI * 180;
        return AngularCoord;
    }
}
