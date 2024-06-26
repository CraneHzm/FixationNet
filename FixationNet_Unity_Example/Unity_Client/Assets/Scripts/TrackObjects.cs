﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;


// Track the positions of the task-related objects for the eye camera
public class TrackObjects : MonoBehaviour
{    
    public class ObjectInfo
    {
        public string objectName = "";
        public Vector3 position = Vector3.zero;
        public float distance = 0;
    }

    // track task-related objects in the field of view of the eye camera
    public Camera eyeCamera;
    // the number of task-related objects to track
    public int objectNumber = 3;
    // save all the task-related objects in this list
    List<Transform> taskObjects = new List<Transform>();
    // save the tracked task-related objects
    List<ObjectInfo> trackedObjects = new List<ObjectInfo>();
    public string trackedObjectsString;


    void Awake()
    {
        foreach (Transform child in this.transform)
            taskObjects.Add(child);
        Debug.Log("Task-Related Object Number: " + taskObjects.Count);
        trackedObjects = new List<ObjectInfo>(objectNumber);
        for (int i = 0; i < objectNumber; ++i)
        {
            ObjectInfo info = new ObjectInfo();   
            trackedObjects.Add(info);
        }
            
        Debug.Log("Tracked Object Number: " + trackedObjects.Count);        
        for (int i = 0; i < objectNumber; ++i)
        {
            // Object Type: 0 for NONE (no object), 1 for Target, and 2 for Distractor.
            int objectType = 0;
            Vector3 position = Vector3.zero;
            float distance = 0;
            if (i == 0)
                trackedObjectsString = objectType.ToString() + "," + position.x.ToString("f2") + "," + position.y.ToString("f2") + "," + distance.ToString("f2");
            else
                trackedObjectsString += "," + objectType.ToString() + "," + position.x.ToString("f2") + "," + position.y.ToString("f2") + "," + distance.ToString("f2");
        }
    }

    // Update is called once per frame
    void Update()
    {
        // clear the tracked objects
        int trackedObjectNumber = 0;        

        List<ObjectInfo> taskObjectsInfos = new List<ObjectInfo>(taskObjects.Count);
        // track task-related objects for the eye camera
        foreach (Transform taskObject in taskObjects)
        {
            ObjectInfo info = new ObjectInfo();
            info.objectName = taskObject.name;
            Vector3 position = eyeCamera.WorldToScreenPoint(taskObject.position);
            info.position = position;
            // distance between eye and the object
            info.distance = Mathf.Sign(position.z) * (taskObject.position - eyeCamera.transform.position).magnitude;            
            taskObjectsInfos.Add(info);
        }
        // record a given number of nearest objects in the camera's field of view
        taskObjectsInfos.Sort((info1, info2) => info1.distance.CompareTo(info2.distance));
        foreach (ObjectInfo info in taskObjectsInfos)
        {
            if (trackedObjectNumber >= objectNumber)
                break;

            if (info.position.x >= 0 && info.position.x <= eyeCamera.pixelWidth && info.position.y >= 0 && info.position.y <= eyeCamera.pixelHeight 
                && info.position.z >= eyeCamera.nearClipPlane && info.position.z <= eyeCamera.farClipPlane)
            {
                Ray ray = eyeCamera.ScreenPointToRay(info.position);
                RaycastHit hitInfo;
                bool rcHit = Physics.Raycast(ray, out hitInfo, info.distance);
                // If there is no collider between eye camera and the object
                if (!rcHit)
                {
                    //Debug.Log("no collider");
                    trackedObjects[trackedObjectNumber].objectName = info.objectName;                    
                    // DGaze Model uses angular coordinates as inputs.
                    Coordinate screenCoord;
                    // (0, 0) at Bottom-left, (1, 1) at Top-right
                    screenCoord.posX = info.position.x / eyeCamera.pixelWidth;
                    screenCoord.posY = info.position.y / eyeCamera.pixelHeight;
                    //Debug.Log("Screen Coord: " + screenCoord.posX + ", " + screenCoord.posY);
                    Coordinate angularCoord = ScreenCoord2AngularCoord(screenCoord);
                    trackedObjects[trackedObjectNumber].position.x = angularCoord.posX;
                    trackedObjects[trackedObjectNumber].position.y = angularCoord.posY;
                    //Debug.Log("Angular Coord: " + angularCoord.posX + ", " + angularCoord.posY);
                    
                    trackedObjects[trackedObjectNumber].position.z = info.position.z;
                    trackedObjects[trackedObjectNumber].distance = info.distance;
                    trackedObjectNumber++;
                }
                else
                {
                    // if the collider is the object itself
                    if (hitInfo.collider.name == info.objectName)
                    {
                        //Debug.Log("collider");
                        trackedObjects[trackedObjectNumber].objectName = info.objectName;
                        // DGaze Model uses angular coordinates as inputs.
                        Coordinate screenCoord;
                        // (0, 0) at Bottom-left, (1, 1) at Top-right
                        screenCoord.posX = info.position.x / eyeCamera.pixelWidth;
                        screenCoord.posY = info.position.y / eyeCamera.pixelHeight;
                        //Debug.Log("Screen Coord: " + screenCoord.posX + ", " + screenCoord.posY);
                        Coordinate angularCoord = ScreenCoord2AngularCoord(screenCoord);
                        trackedObjects[trackedObjectNumber].position.x = angularCoord.posX;
                        trackedObjects[trackedObjectNumber].position.y = angularCoord.posY;
                        //Debug.Log("Angular Coord: " + angularCoord.posX + ", " + angularCoord.posY);

                        trackedObjects[trackedObjectNumber].position.z = info.position.z;
                        trackedObjects[trackedObjectNumber].distance = info.distance;
                        trackedObjectNumber++;
                    }
                }
            }
        }
        
        for (int i = trackedObjectNumber; i < objectNumber; ++i)
            trackedObjects[i] = new ObjectInfo();

        // save the tracked object info
        string infos = null;
        for (int i = 0; i < objectNumber; ++i)
        {
            //Debug.Log(trackedObjects[i].objectName);
            string name = trackedObjects[i].objectName.Split('_')[0];
            //Debug.Log(name);
            // Object Type: 0 for NONE (no object), 1 for Target, and 2 for Distractor. 
            int objectType = 0;
            //Targets and Distractors in a Visual Search Task.
            if (name == "Target")
                objectType = 1;
            if (name == "Distractor")
                objectType = 2;

            Vector3 position = trackedObjects[i].position;
            float distance = trackedObjects[i].distance;
            if (i == 0)
                infos = objectType.ToString() + "," + position.x.ToString("f2") + "," + position.y.ToString("f2") + "," + distance.ToString("f2");
            else
                infos += "," + objectType.ToString() + "," + position.x.ToString("f2") + "," + position.y.ToString("f2") + "," + distance.ToString("f2");
        }
        trackedObjectsString = infos;
        //Debug.Log("trackedObjectsString: " + trackedObjectsString);
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
        AngularCoord.posX = Mathf.Atan((ScreenCoord.posX - ScreenCenterX) / ScreenDist)/ Mathf.PI*180;
        AngularCoord.posY = Mathf.Atan((ScreenCoord.posY - ScreenCenterY) / ScreenDist) / Mathf.PI * 180;
        return AngularCoord;
    }

}
