import gc


import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import matplotlib as plt 
import numpy as np 
import tensorflow as tf 
import tensorflow_datasets as tfds
from typing import NamedTuple, Optional, Sequence 
import haiku as hk
from sgld_utils import (
    SGLDConfig,
    run_sgld
)
from utils import to_json_friendly_tree, running_mean
import optax
from sacred import Experiment
# Create a new expriment
ex= Experiment('llc_curve')

class TrainingConfig(NamedTuple):
    optim: str 
    learning_rate: float
    batch_size: int 
    num_steps: int 
    momentum: float = None
    l2_regularization: float = None


class CustomResNet18(hk.nets.ResNet):
    def __init__(
            self,
            num_classes: int,
            k: int = 64,
            name: Optional[str] = None,
            strides: Sequence[str] = (1,2,2,2),
    ):
        custom_configs = {
            "blocks_per_groups": (2,2,2,2),
            "bottleneck": False,
            "channels_per_group": (k,2 * k, 4 * k, 8 *k),
            "use_projection": (False, True, True,True),
        }
        super().__init__(num_classes=num_classes,
                         bn_config=None,
                         initial_conv_config={"output_channels": k, "kernel_shape": 7, "stride": 2, "padding": "SAME"},
                         resnet_v2=False,
                         strides=strides,
                         logits_config=None,
                         name=name,
                         **custom_configs)
def make_resnet18(num_classes=10, k=64):
    def net_fn(x, is_training=True):
        model = CustomResNet18(num_classes=num_classes,k=k)
        return model
    return hk.transform_with_state(net_fn)

def introduce_label_noise(labels, noise_level):
    n_samples= len(labels)
    n_noisy= int(n_samples * noise_level)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    min_Label = labels.min()
    max_label = labels.max()
    new_labels = np.random.randint(min_Label, max_label+1, n_noisy)
    noisy_labels = labels.copy()
    noisy_labels[noisy_indices] = new_labels

    return noisy_labels

def load_cifar10(noise_level=None):
    ds_builder = tfds.builder('cifar10')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1, shuffle_files=False))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_images, train_labels = train_ds['image'], train_ds['label']
    test_images, test_labels = test_ds['images'], test_ds['label']   

    train_images = train_images.astype(jnp.float32) / 255.0
    test_images = test_images.astype(jnp.float32) / 255.0 

    datasets = {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

    if noise_level is not None and noise_level > 0.0:
          datasets['noisy_train_labels'] =  introduce_label_noise(train_labels, noise_level)
    return datasets

def initialize_model(rng, num_classes=10, k=64):
    model = make_resnet18(num_classes=num_classes, k=k)
    dummy_input = jnp.ones([1,32,32,3], jnp.float32)
    params, state= model.init(rng,dummy_input, True)
    return model, params, state

def batch_generator(x,y, batch_size, rngkey):
    num_examples= len(x)
    while True:
        rngkey, _= jax.random.split(rngkey)
        perm = jax.random.permutation(rngkey, jnp.arange(num_examples))
        for i in range(0, num_examples, batch_size):
            batch_indices = perm[i:i+batch_size]
            yield x[batch_indices], y[batch_indices]
        rngkey, _ = jax.random.split(rngkey)

def evaluate_accuracy(model, params, state, x, y, rngkey):
    logits, _ = model.apply(params, state, rngkey,x, False)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions ==y)

def logit_logit_cross_entropy(logits1, logits2):
    probs1 = jax.nn.softmax(logits1, axis=-1)
    return -jnp.sum(probs1 * jax.nn.log_softmax(logits2, axis=-1), axis=-1)

@ex.config
def cfg():
    expt_name = None
    sgld_config = {
        'epsilon': 5e-6,
        'gamma': 1.0,
        'num_steps': 1000,
        "num_chains": 1,
        "batch_size": 1024
    }
    loss_trace_minibatch = True
    model_data_config = {
        "model_name": "resnet18",
        "data_name": "cifar10",
        "label_noise_level": None,
        "layer_width_factor": 64,
    }
    training_config = {
        "optim": "sgd",
        "learning_rate": 1e-3,
        "momentum": None,
        "batch_size": 128,
        "num_steps": 20000,
        "l2_regularization":None,
    }
    force_realisable = False 
    seed = 42
    logging_period = 200 
    verbose = False
    do_plot = False
@ex.automain
def run_expriment(
        _run,
        expt_name,
        sgld_config,
        loss_trace_minibatch,
        model_data_config,
        training_config,
        force_realisable,
        seed,
        logging_period,
        verbose,
        do_plot,
):
    #seeding
    np.random.seed()
    rngkey = jax.random.PRNGKey(seed)
    tf.random.set_seed(seed)

    sgld_config = SGLDConfig(**sgld_config)
    training_config= TrainingConfig(**training_config)
    label_noise_level = model_data_config['label_noise_level']
    dataset = load_cifar10(noise_level=label_noise_level)
    x_train, x_test, y_test= dataset['train_images'], dataset['train_labels'], dataset['test_labels']
    using_label_noise = label_noise_level is not None and label_noise_level > 0.0 
    if using_label_noise:
        y_train = dataset['noisy_train_labels'] 
        y_train_no_noise = dataset['train_labels']
    else:
        y_train = dataset["train_labels"]


    layer_width_factor = model_data_config['layer_width_factor']
    rngkey, subkey = jax.random.split(rngkey)
    model, trained_param, model_state = initialize_model(rngkey,num_classes=10, k=layer_width_factor)
    param_count = sum(np.prod(p.shape) for p in jtree.tree_leaves(trained_param))
    print(f"Total number of parameters: {param_count}")


    rngkey, subkey= jax.random.split(rngkey)
    train_dataset_iter = batch_generator(
        x_train,
        y_train,
        training_config.batch_size,
        rngkey
    )
    rngkey, subkey = jax.random.split(rngkey)
    num_training_data = x_train.shape[0]

    if training_config.optim.lower() == "sgd":
        optimizer = optax.adam(
            learning_rate= training_config.learning_rate,
            momentum=training_config.momentum
        )
    elif training_config.optim.lower() == "adam":
            optimizer = optax.adam(
            learning_rate= training_config.learning_rate,)
    else:
        raise ValueError("Unkown optimizer")
    opt_state= optimizer.init(trained_param)

    def compute_loss(params, state, rngkey, x,y, is_training, l2_regularization=0.0):
        labels_one_hot = jax.nn.one_hot(y,10)
        logits, new_state = model.apply(params, state, rngkey, x, is_training)
        loss_val = jnp.mean(optax.softmax_cross_entropy(logits, labels=labels_one_hot))
        if l2_regularization is not None and l2_regularization > 0.0:
            l2_loss = l2_regularization* sum(jnp.sum(jnp.square(p)) for p in jtree.tree_leaves(params))
            loss_val += l2_loss
        return loss_val, new_state

    training_loss_fn = lambda parameter, state, rengkey, x, y:compute_loss(
        parameter,
        state,
        rngkey,
        x,
        y,
        is_training=True,
        l2_regularization=training_config.l2_regularization
    )
    @jax.jit
    def update_step(params, state, rngkey, x,y,opt_state):
        (loss_val, new_state), grad= jax.value_and_grad(training_loss_fn, has_aux=True)(params,state,rngkey,x,y)
        updates, new_opt_state= optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params,updates)
        return loss_val, new_params, new_state, new_opt_state

    if force_realisable:
        @jax.jit
        def sgld_outer_loss_fn(parameter, model_state, rngkey, x, y):
            model_output_logit = model.apply(parameter, model_state, rngkey, x, False)[0]
            return jnp.mean(logit_logit_cross_entropy(y, model_output_logit))
    else:
        @jax.jit
        def sgld_outer_loss_fn(parameter, model_state, rngkey, x,y):
            return compute_loss(parameter,model_state,rngkey,x,y, is_training=False, l2_regularization=None)[0]
        
        
    # Train the model and do SGLD at some interval

    itemp = 1/ np.log(num_training_data)
    max_steps = training_config.num_steps

    _run.info = []
    t = 0 

    while t < max_steps:
        x_batch, y_batch = next(train_dataset_iter)
        rngkey, subkey = jax.random.split(rngkey)
        train_loss, trained_param, model_state, opt_state = update_step(
            trained_param,
            model_state,
            subkey,
            x_batch,
            y_batch,
            opt_state
        )
        if t % logging_period == 0 or t == max_steps - 1:
            gc.collect()
            if force_realisable:
                y= model.apply(trained_param, model_state, rngkey, x_train, False)[0]
            else:
                if using_label_noise:
                    y = y_train_no_noise
                else:
                    y = y_train
            
            loss_fn = lambda params, state, rngkey, x, y: sgld_outer_loss_fn(params, state, rngkey, x,y)
            loss_trace, distances,acceptance_probs = run_sgld(
                subkey,
                sgld_config,
                loss_fn,
                trained_param,
                x_train,
                y,
                itemp=itemp,
                trace_batch_loss=loss_trace_minibatch,
                compute_distances=False,
                compute_mala_acceptance_probs=False,
                verbose=verbose
            )
            init_loss = loss_fn(trained_param, model_state,x_train, y)
            lamddahat = float(np.mean(loss_trace)-init_loss)* num_training_data*itemp
            print(f"Step {t}: Loss: {train_loss:.4f}, Lambda_hat: {lamddahat:.4f}")
            test_loss = compute_loss(trained_param, model_state, rngkey,x_test, y_test, is_training=False, l2_regularization=training_config.l2_regularization)[0]
            test_loss_no_reg = compute_loss(trained_param, model_state, rngkey,x_test, y_test, is_training=False, l2_regularization=None)[0]
            print(f"Step {t}: Test Loss: {test_loss:.4f}, Test Loss (no reg): {test_loss_no_reg:.4f}")
            rngkey, subkey = jax.random.split(rngkey)
            train_acc = evaluate_accuracy(model, trained_param, model_state, x_train, y_train, subkey)
            rngkey, subkey = jax.random.split(rngkey)
            test_acc = evaluate_accuracy(model, trained_param, model_state, x_test, y_test, subkey)
            rec = {
                "t": t+1 ,
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
                "test_loss_no_reg": float(test_loss_no_reg),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "lambda_hat": float(lamddahat),
                "loss_trace": loss_trace,
                "init_loss": float(init_loss),
                "slgld_distances": distances,
                "sgld_acceptance_probs": acceptance_probs,
            }
            if verbose:
                print(f"Step {t}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Loss (no reg): {test_loss_no_reg:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Lambda_hat: {lamddahat:.4f}")

            _run.info.append(to_json_friendly_tree(rec))
        t += 1
        if t >= max_steps:
            break   
    if do_plot:
        df = pd.DataFrame(_run.info)
        fig, ax = plt.subplots()
        ax.plot(df['t'], df['train_loss'], label='Train Loss')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')

        ax = ax.twinx()
        title = (
            f"$T={training_config.num_steps}$, " 
            f"$k={logging_period}$, "
            f"RNG seed$={seed}$"
        )
        ax.plot(df["t"], np.clip(df["lambdahat"], a_min=0, a_max=np.inf), "kx", alpha=0.3)
        yvals = running_mean(df["lambdahat"])
        ax.plot(df["t"], yvals, "g-")
        ax.set_ylabel("Estimated LLC, $\hat{\lambda}(w^*)$")
        ax.set_title(title, fontsize="large")
        filepath = f"llc_curve_plot_{expt_name}_{seed}.pdf"
        fig.savefig(filepath, bbox_inches="tight")
        print(f"Image saved at: {filepath}")
    return
