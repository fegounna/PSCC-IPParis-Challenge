MODEL = os.environ.get("MODEL")

if __name__ == "__main__":

    model = dispatcher.MODELS[MODEL]
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoint = torch.load(f"models/checkpoint_{MODEL}.pht.tar", map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    num_batches = 1
    test_files = get_test_files()
    test_ds = monai.data.Dataset(data=test_files,transform=orig_transforms)
    test_ldr = DataLoader(test_ds,batch_size=num_batches)
    tk0 = tqdm(test_ldr, total=num_batches)

    model.eval()


    save_directory = f'\predictions\{MODEL}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with torch.no_grad():
        for data in tk0:
            inputs = data["image"]
            inputs = inputs.to(device, dtype=torch.float)
            output = model(inputs)

            for prediction,id in zip(output,data["id"]):
                prediction = prediction[0]
                output_np = prediction.cpu().detach().numpy()
                threshold_value = 0.5
                thresholded_output = output_np > threshold_value
                thresholded_output = thresholded_output.astype(np.uint8)
                voxel_size = [0.9765625, 0.9765625, 3.0]
                affine = np.diag(voxel_size + [1])
                nifti_img = nib.Nifti1Image(thresholded_output, affine)
                path = os.path.join(save_directory, f"""{id}.nii.gz""")
                nifti_img.to_filename(path)
        
        tk0.close()
import sys
sys.path.append(r'C:\Users\Moakher\PSCC_datachallenge')
from hackathon.submission_gen import submission_gen
csvpath = r"C:\Users\Moakher\Desktop\hackaton\output.csv"
result = submission_gen(save_directory, csvpath)
print(result)