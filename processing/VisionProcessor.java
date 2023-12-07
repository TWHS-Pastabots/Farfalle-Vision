package processing;

import java.util.List;
import org.photonvision.PhotonCamera;

public class VisionProcessor {

    private PhotonCamera camera1;
    private PhotonPipelineResult camera1Pipeline;
    private List<PhotonTrackedTarget> targets;
    private PhotonTrackedTarget currentTarget;

    public VisionProcessor() {}

    public void init() {
        camera1 = new PhotonCamera("camera_1");
    }

    public void reloadPipeline () {
        camera1Pipeline = camera1.getLatestResult();
    }

    public int getTargetID() {
        
        if(camera1Pipeline.hasTargets()) {
            targets = camera1Pipeline.getTargets();
            currentTarget = targets.getBestTarget();
            return currentTarget.getFuducialID();
        }
        //not an actual AprilTag value
        return 9;
    }

    public int getDistanceToTag() {
    }
}