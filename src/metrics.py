from monai.metrics import DiceMetric
import nibabel as nib
import cv2

def find_largest_containing_circle(segmentation, pixdim=(0.9765625, 0.9765625, 3.0)):
    largest_circle = None
    largest_slice = -1
    max_radius = -1

    segmentation8 = segmentation.astype(np.float32).astype('uint8')
    for i in range(segmentation8.shape[-1]):
        # Find the contours in the segmentation
        contours, _ = cv2.findContours(image = segmentation8[0,:,:,i], mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Fit the smallest circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius > max_radius:
                max_radius = radius
                largest_circle = ((int(x), int(y)), int(radius))
                largest_slice = i
    recist = max_radius * 2 * pixdim[0]
#     print(max_radius)
    predicted_volume = np.round(np.sum(segmentation.flatten())*pixdim[0]*pixdim[1]*pixdim[2]*0.001,2)
    return recist, predicted_volume, largest_circle, largest_slice

def dice_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    N = y_pred.size(0)
    y_pred_f = (y_pred > thr).view(N, -1)
    y_true_f = y_true.view(N, -1)
    
    inter = (y_pred_f*y_true_f).sum(1)
    den = y_pred_f.sum(1) + y_true_f.sum(1)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean()
    return dice


def score(true, predicted):
    dice_score = dice_coef(true, predicted).item()
    recist_scores = []
    volume_scores = []
    for true_seg, predicted_seg in zip(true,predicted):
        true_recist, true_volume,_,_ = find_largest_containing_circle(true_seg)
        predicted_recist, predicted_volume,_,_ = find_largest_containing_circle(predicted_seg)
        recist_scores.append(np.abs(predicted_recist-true_recist))
        volume_scores.append(np.abs(predicted_volume-true_volume))
    score = np.array([1-dice_score,np.mean(recist_scores),np.mean(volume_scores)])
    return score

def metric_score(dataset, data_loader, model):
    model.eval()
    final_score = np.array([0.,0.,0.])
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)
    with torch.no_grad():
        for data in tk0:
            inputs = data["image"]
            targets = data["label"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            output = model(inputs)
            final_score += score(targets,output)
    tk0.close()
    return  final_score / num_batches