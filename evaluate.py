import numpy as np
import matplotlib.pyplot as plt


def plot_roc(y_valid, y_predicted):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_valid, y_predicted)
    auc_score = auc(fpr, tpr)
    # Zoom in view of the upper left corner.
    plt.figure(figsize = (9,6))
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--', lw = 2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(left = 0, right = 1)
    plt.ylim(top = 1, bottom = 0)
    return auc_score


def plot_layer(model, layer_idx, img, cmap = 'viridis', normalize = False):
    h,w,d = np.shape(img)
    img = img * 255
    m = Model(inputs = model.input, outputs = model.get_layer(index = layer_idx).output)
    p = m.predict(np.reshape(img, (1,h,w,d)))
    img_stack = []
    p_size = np.shape(p)[3]
    for i in range(p_size):
        p_out = p[0,:,:,i]
        img_stack.append(p_out)
    grid_plot(img_stack) 
    return


def grid_plot(img_stack, outfile = 'grid_plot.png'):
    n_imgs = len(img_stack)
    nrows = int(np.floor(np.sqrt(n_imgs)))
    ncols = int(np.ceil(n_imgs / nrows))
    F = plt.figure(figsize = (35,35))
    F.subplots_adjust(left = 0.05, right = 0.95)
    grid = ImageGrid(F, 142, nrows_ncols = (nrows, ncols), axes_pad = 0.0, share_all = True,
                     label_mode = "L", cbar_location = "top", cbar_mode = "single")
    i = 0
    for img in img_stack:
        im = grid[i].imshow(img, interpolation = "nearest", vmin = 0, vmax = 255)
        i += 1 
    grid.cbar_axes[0].colorbar(im)
    plt.savefig(outfile)
    for cax in grid.cbar_axes:
        cax.toggle_label(True)
    return
