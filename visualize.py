import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import traceback

ARC_COLORMAP = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF']
)
ARC_NORM = colors.Normalize(vmin=0, vmax=10)

def visualize_task(
    task_data,
    task_solutions=None,
    n_test=0,
    title="ARC Task",
    figsize=(6, 3),
    save=False,
    grid=True,
    answer=None,
    file='eval',
    prob_grid=None,
    key=None,
    score=None,
):
    # if competition_run:
    #     return

    if answer is None:
        answer = []

    try:
        if file == 'eval':
            file = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"
        # elif file == 'train':
        #     file = file.replace('evaluation','training')
        with open("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json" if not file else file, 'r') as f:
            test = json.load(f)
            
        with open("/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json", 'r') as f:
            test.update(json.load(f))

        if score is not None and key:
            if file:
                sol_file = file.replace('test','training').replace('challenges','solutions')
            with open("/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json" if not file else sol_file, 'r') as f:
                answer = json.load(f).get(key.split('_')[0],[])
                ind = int(key.split('_')[-1])
                if answer:
                    try:
                        answer = answer[ind]
                    except:
                        answer = test.get(key.split('_')[0])['train'][9-ind]['output']
            
        if isinstance(task_data,str):
            task_id = task_data
            title = task_id + ' ' + title
            task_data = test.get(task_data,{})
    
            if file:
                sol_file = file.replace('test','training').replace('challenges','solutions')
            with open("/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json" if not file else sol_file, 'r') as f:
                solutions = json.load(f).get(task_id,[])
            if task_solutions is None:
                task_solutions = solutions
            else:
                answer = solutions
                
    except:
        print('Load in vis failed.')
        try:
            print(traceback.format_exc())
        except:
            pass
        return

    if prob_grid is not None:
        fig, axs = plt.subplots(2,3,figsize=(10,4))
        key=title.split('_')[0]
        try:
            test_input = test[key]['test'][int(title.split('.')[0][-1])]['input']
        except:
            test_input = test[key]['train'][9-int(title.split('.')[0][-1])]['input']
        task_data = np.array(task_data)
        prob_grid = np.array(prob_grid)
        prob_grid_exp = np.exp(prob_grid)
        # prob_grid_linrescale = (prob_grid - prob_grid.min()) / (prob_grid.max() - prob_grid.min())
        # prob_grid_rescale = (prob_grid_exp - prob_grid_exp.min()) / (prob_grid_exp.max() - prob_grid_exp.min())
        plt.suptitle(title, fontsize=8)

        task_data_shape = task_data.shape
        x,y = task_data_shape
        num_rot = title.count('rot90')
        if num_rot in [1,3] or 'transpose' in title:
            if num_rot in [1,3] and 'transpose' in title:
                direction_grid = np.arange(x*y).reshape(x,y)
            else:
                direction_grid = np.arange(x*y).reshape(y,x)
        else:
            direction_grid = np.arange(x*y).reshape(x,y)
        if 'transpose' in title:
            direction_grid = np.swapaxes(direction_grid, 0, 1)
        direction_grid = np.rot90(direction_grid,k=-num_rot)
        
    elif score is not None:
        if key:
            r = (len(score)+1)//4+1
            try:
                test_input = test[key.split('_')[0]]['test'][int(key.split('.')[0][-1])]['input']
            except:
                test_input = test[key.split('_')[0]]['train'][9-int(key.split('.')[0][-1])]['input']
        else:
            r = len(score)//4+1
        fig, axs = plt.subplots(r, 4, figsize=(10,2*r))
        
        
    else:
        # Extract examples
        train_examples = task_data.get('train', [])
        test_examples = task_data.get('test', [])
        original_test = task_data.get('original_test', [])
        trinf_examples = task_data.get('trinf_test', [])
    
        # Concatenate test examples in preferred order
        test_examples = trinf_examples + test_examples + original_test
    
        # Adjust task_solutions if TRINF examples are present
        if trinf_examples and task_solutions:
            task_solutions = [None] + task_solutions
    
        num_train = len(train_examples)
        num_test = len(test_examples)
        cols = num_train + num_test + (1 if answer else 0)
    
        # Resize if gridlines are requested
        if grid:
            figsize = (int(figsize[0] * 1.5), int(figsize[1] * 1.5))
    
        fig, axs = plt.subplots(2, cols, figsize=figsize)
        plt.suptitle(title, fontsize=12)
    fontsize = 8

    def draw_grid(ax, grid_data, title_str, prob_grid=None, inv=False, alpha=False):
        # ax.imshow(grid_data, cmap=ARC_COLORMAP, norm=ARC_NORM)
        height, width = len(grid_data), len(grid_data[0])
        ax.set_title(f"{title_str}({height},{width})", fontsize=fontsize)

        grid_data = np.array(grid_data)
        grid_data[grid_data==-1] = 10

        if alpha:
            # Draw base grid
            base_im = ax.imshow(grid_data, cmap=ARC_COLORMAP, norm=ARC_NORM)
        
            if prob_grid is not None:
                # Normalize alpha: high prob = low opacity
                if inv:
                    prob_grid = 1.0 - prob_grid  # inverse
                alpha_mask = np.clip(prob_grid, 0.0, 1.0)

                base = ax.imshow(np.ones(grid_data.shape,dtype=int)*10, cmap=ARC_COLORMAP, norm=ARC_NORM)
                overlay = ax.imshow(grid_data, cmap=ARC_COLORMAP, norm=ARC_NORM, alpha=alpha_mask)

        else:
            try:
                # Convert color-mapped indices to RGB image
                rgba = ARC_COLORMAP(ARC_NORM(grid_data))[:, :, :3]  # shape (H, W, 3)
            except:
                print(grid_data)
        
            # Apply brightening using prob_grid
            if prob_grid is not None:
                # print('rgba_before',rgba)
                prob_grid = np.clip(prob_grid, 0, 1)
                prob_grid *= 0.8
                # prob_grid = (prob_grid - prob_grid.min()) / (prob_grid.max() - prob_grid.min())
                if inv:
                    prob_grid = 1-prob_grid
                prob_grid = prob_grid[:, :, np.newaxis]  # shape (H, W, 1)
                # print('prob_grid newaxis',prob_grid)
                rgba = rgba * (1 - prob_grid) + prob_grid * 1.0  # brightening toward white
                # print('rgba_after',rgba)
        
            # Show the modified RGB image
            ax.imshow(rgba)
            
        if grid:
            ax.set_xticks(np.arange(width) - 0.5)
            ax.set_yticks(np.arange(height) - 0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, color='white' if prob_grid is None else 'black', linewidth=0.2)
        else:
            ax.axis('off')

    if prob_grid is not None:
        draw_grid(axs[0,0], test_input, 'Test input')
        draw_grid(axs[1,0], answer, 'Target')
        
        def draw_prob_heatmap(ax, prob_grid, title_str="Probability Heatmap", fontsize=fontsize):
            im = ax.imshow(prob_grid, cmap='viridis')
            height, width = len(prob_grid), len(prob_grid[0])
            ax.set_title(title_str, fontsize=fontsize)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if grid:
                ax.set_xticks(np.arange(width) - 0.5)
                ax.set_yticks(np.arange(height) - 0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.grid(True, color='black', linewidth=0.2)
            else:
                ax.axis('off')
                
            # Add text annotations from direction_grid if provided
            if direction_grid is not None:
                # Normalize colormap to match imshow
                norm = im.norm
                cmap = im.cmap
        
                for i in range(height):
                    for j in range(width):
                        val = direction_grid[i][j]
                        if val in [0, 1]:
                            # Get background color from colormap
                            bg_color = cmap(norm(prob_grid[i][j]))
                            # Compute luminance (perceived brightness)
                            luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                            text_color = 'black' if luminance > 0.5 else 'white'
        
                            ax.text(j, i, str(val), ha='center', va='center', color=text_color, fontsize=fontsize - 2)

        draw_grid(axs[0,1], task_data, 'Predict')
        draw_prob_heatmap(axs[0,2],prob_grid,title_str='logsoftmax')
        draw_grid(axs[1,1], task_data, 'Predict Prob', prob_grid=prob_grid_exp)
        draw_prob_heatmap(axs[1,2],prob_grid_exp,title_str='prob')
        
    elif score is not None:
        if answer:
            score = ['Answer'] + score
            task_data = [answer] + task_data
        if key:
            score = ['Input'] + score
            task_data = [test_input] + task_data
        for i, ax in enumerate(axs.flat):
            try:
                draw_grid(ax, task_data[i], score[i])
            except:
                ax.axis('off')
    else:
        # Plot training inputs and outputs
        for i, example in enumerate(train_examples):
            draw_grid(axs[0, i], example['input'], "Train Input")
            draw_grid(axs[1, i], example['output'], "Train Output")
    
        # Plot test inputs and predicted or actual outputs
        for i, example in enumerate(test_examples):
            col_idx = num_train + i
            label = "TRINF Input" if (trinf_examples or original_test) and i == 0 else "Test Input"
            draw_grid(axs[0, col_idx], example['input'], label)
    
            try:
                if 'output' in example:
                    draw_grid(axs[1, col_idx], example['output'], "Test Output")
                elif task_solutions:
                    draw_grid(
                        axs[1, col_idx],
                        task_solutions[i],
                        "TRINF Output" if (trinf_examples or original_test) and i == 0 else "Test Output"
                    )
                else:
                    axs[1, col_idx].set_title("Test Output: ?", fontsize=fontsize)
                    axs[1, col_idx].axis('off')
            except Exception:
                axs[1, col_idx].set_title("Test Output: ?", fontsize=fontsize)
                axs[1, col_idx].axis('off')
    
        # Plot provided answer if any
        if answer:
            col_idx = num_train + num_test
            axs[0, col_idx].axis('off')
            draw_grid(axs[1, col_idx], answer[task_solutions.count(None)], "Answer")

    plt.tight_layout()
    if save:
        plt.savefig(f"{save}.png", bbox_inches="tight")
    plt.show()
