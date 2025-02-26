"""Train an MLP on MNIST on one random seed. Serialize the model for
interpolation downstream."""
import argparse
import os

import augmax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, value_and_grad, vmap
from jax.tree_util import tree_map
from tqdm import tqdm

from utils import ec2_get_instance_type, flatten_params, rngmix, timeblock
tfds.core.utils.gcs_utils._is_gcs_disabled = True
# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

activation = nn.relu


class MLPModel(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (-1, 28 * 28))
        x = nn.Dense(512)(x)
        x = activation(x)
        x = nn.Dense(512)(x)
        x = activation(x)
        x = nn.Dense(512)(x)
        x = activation(x)
        x = nn.Dense(10)(x)
        x = nn.log_softmax(x)
        return x

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)


def make_stuff(model):
    normalize_transform = augmax.ByteToFloat()

    @jit
    def batch_eval(params, images_u8, labels):
        images_f32 = vmap(normalize_transform)(None, images_u8)
        logits = model.apply({"params": params}, images_f32)
        y_onehot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
        num_correct = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(y_onehot, axis=-1))
        return loss, {"num_correct": num_correct}

    @jit
    def step(train_state, images_f32, labels):
        (l, info), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, images_f32, labels)
        # for k in g.keys():
        #     tf.summary.histogram(k+'/kernels',g[k]['kernel'])   
        #     tf.summary.histogram(k+'/bias',g[k]['bias'])      
        return train_state.apply_gradients(grads=g), {"batch_loss": l, **info}

    def dataset_loss_and_accuracy(params, dataset, batch_size: int):
        num_examples = dataset["images_u8"].shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
        # Can't use vmap or run in a single batch since that overloads GPU memory.
        losses, infos = zip(*[
            batch_eval(
                params,
                dataset["images_u8"][batch_ix[i, :], :, :, :],
                dataset["labels"][batch_ix[i, :]],
            ) for i in range(num_batches)
        ])
        return (
            jnp.sum(batch_size * jnp.array(losses)) / num_examples,
            sum(x["num_correct"] for x in infos) / num_examples,
            # params,
        )

    return {
        "normalize_transform": normalize_transform,
        "batch_eval": batch_eval,
        "step": step,
        "dataset_loss_and_accuracy": dataset_loss_and_accuracy
    }


def load_mnist_datasets():
    """Return the training and test datasets, unbatched."""
    # See https://www.tensorflow.org/datasets/overview#as_batched_tftensor_batch_size-1.
    train_ds_images_u8, train_ds_labels = tfds.as_numpy(
        tfds.load("mnist", split="train", batch_size=-1, as_supervised=True))
    # tfds.load("mnist_corrupted", split="train", batch_size=-1, as_supervised=True))

    test_ds_images_u8, test_ds_labels = tfds.as_numpy(
        tfds.load("mnist", split="test", batch_size=-1, as_supervised=True))
    # tfds.load("mnist_corrupted", split="test", batch_size=-1, as_supervised=True))

    train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
    test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
    return train_ds, test_ds




def load_mnistc_datasets(noise_type):
    train_ds_images_u8, train_ds_labels = tfds.as_numpy(
        # tfds.load("mnist", split="train", batch_size=-1, as_supervised=True))
        tfds.load(f"mnist_corrupted/{noise_type}", split="train", batch_size=-1, as_supervised=True))

    test_ds_images_u8, test_ds_labels = tfds.as_numpy(
        # tfds.load("mnist", split="test", batch_size=-1, as_supervised=True))
        tfds.load(f"mnist_corrupted/{noise_type}", split="test", batch_size=-1, as_supervised=True))

    train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
    test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
    return train_ds, test_ds

def shuffle_grouping(num_samples,num_group):
    assert (num_samples%num_group)==0, "Sample num should be divisible by group num!"  
    num_per_group=int(num_samples/num_group)
    
    new_inds=np.arange(0,num_samples,1)
    np.random.default_rng(123).shuffle(new_inds)
    grouped_index=new_inds.reshape([num_group,num_per_group])
        
    return grouped_index

def load_merged_datasets(noise_types):
    trains={}
    tests={}


    for n in noise_types:
        trains[n],tests[n]=load_mnistc_datasets(n)
    
    train_len=len(trains[noise_types[0]]["images_u8"])
    test_len=len(tests[noise_types[0]]["images_u8"])

    train_noise_sampler = shuffle_grouping(train_len,len(noise_types))
    test_noise_sampler = shuffle_grouping(test_len,len(noise_types))


    train_images_u8=trains[noise_types[0]]["images_u8"]
    train_labels= trains[noise_types[0]]["labels"]
    test_images_u8=tests[noise_types[0]]["images_u8"]
    test_labels= tests[noise_types[0]]["labels"]
    # flags = np.array([0]*len(train1["images_u8"])+[1]*len(train2["images_u8"]))
    # ind_ori=perm_train[:np.int(train_len/2)]
    # ind_corr=perm_train[np.int(train_len/2):]
    i=0
    for n in noise_types:
        inds_train=train_noise_sampler[i]
        inds_test=test_noise_sampler[i]
        train_images_u8[inds_train]=trains[n]["images_u8"][inds_train,:,:,:]
        train_labels[inds_train] =trains[n]["labels"][inds_train]

        test_images_u8[inds_test]=tests[n]["images_u8"][inds_test,:,:,:]
        test_labels[inds_test] =tests[n]["labels"][inds_test]
        i=i+1
    train_ds = {
        "images_u8": train_images_u8,
        "labels": train_labels
    }
    test_ds = {
        "images_u8": test_images_u8,
        "labels": test_labels
    }
    return train_ds, test_ds

def load_merged_datasets2():
    train1,test1=load_mnist_datasets()
    train2,test2=load_mnistc_datasets()
    train_len=len(train1["images_u8"])+len(train2["images_u8"])
    test_len=len(test1["images_u8"])+len(test2["images_u8"])
    perm_train = np.random.default_rng(123).permutation(train_len)
    perm_test = np.random.default_rng(123).permutation(test_len)

    train_images_u8=np.concatenate((train1["images_u8"], train2["images_u8"]), axis=0)
    train_labels= np.concatenate((train1["labels"], train2["labels"]), axis=0)
    train_images_u8=train_images_u8[perm_train,:,:,:]
    train_labels=train_labels[perm_train]

    test_images_u8 = np.concatenate((test1["images_u8"], test2["images_u8"]), axis=0)
    test_labels= np.concatenate((test1["labels"], test2["labels"]), axis=0)
    test_images_u8 = test_images_u8[perm_test, :, :, :]
    test_labels = test_labels[perm_test]

    train_ds = {
        "images_u8": train_images_u8,
        "labels": train_labels
    }
    test_ds = {
        "images_u8": test_images_u8,
        "labels": test_labels
    }
    return train_ds, test_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--dataset",type=str,default="mnist",choices=["mnist", "mnist-corrupted", "mnist-merged"])
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=3080)
    parser.add_argument("--noise_type", type=str,nargs='+',default='identity')
    args = parser.parse_args()

    with wandb.init(
            project="git-re-basin",
            entity="lily-le",
            name=args.dataset+"-mlp-"+f"seed{args.seed}",   
            tags=[args.dataset, "mlp", "training",f"seed{args.seed}",f"{args.gpu}"],
            mode="disabled" if args.test else "online",
            job_type="train",
    ) as wandb_run:
        artifact = wandb.Artifact(f"{args.dataset}-mlp-weights-new2", type="model-weights")

        config = wandb.config
        config.ec2_instance_type = ec2_get_instance_type()
        config.test = args.test
        config.seed = args.seed
        config.noise_types=args.noise_type
        config.optimizer = args.optimizer
        config.learning_rate = args.learning_rate
        config.num_epochs = 100
        config.batch_size = 500

        rng = random.PRNGKey(config.seed)

        model = MLPModel()
        stuff = make_stuff(model)

        with timeblock("load_datasets"):
            if args.dataset=="mnist":
                train_ds, test_ds = load_mnist_datasets()
            if args.dataset=="mnist-corrupted":
                assert len(config.noise_type)==1, "noise type should equal to 1"
                train_ds, test_ds = load_mnistc_datasets(config.noise_types[0])
            if args.dataset=="mnist-merged":
                train_ds, test_ds = load_merged_datasets(config.noise_types)
     

            print("train_ds labels hash", hash(np.array(train_ds["labels"]).tobytes()))
            print("test_ds labels hash", hash(np.array(test_ds["labels"]).tobytes()))

            num_train_examples = train_ds["images_u8"].shape[0]
            num_test_examples = test_ds["images_u8"].shape[0]
            assert num_train_examples % config.batch_size == 0
            print("num_train_examples", num_train_examples)
            print("num_test_examples", num_test_examples)

        if config.optimizer == "sgd":
            # See runs:
            # * https://wandb.ai/skainswo/git-re-basin/runs/3blb4uhm
            # * https://wandb.ai/skainswo/git-re-basin/runs/174j7umt
            # * https://wandb.ai/skainswo/git-re-basin/runs/td02y8gg
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=1e-6,
                peak_value=config.learning_rate,
                warmup_steps=10,
                # Confusingly, `decay_steps` is actually the total number of steps,
                # including the warmup.
                decay_steps=config.num_epochs * (num_train_examples // config.batch_size),
            )
            tx = optax.sgd(lr_schedule, momentum=0.9)
        elif config.optimizer == "adam":
            # See runs:
            # - https://wandb.ai/skainswo/git-re-basin/runs/1b1gztfx (trim-fire-575)
            # - https://wandb.ai/skainswo/git-re-basin/runs/1hrmw7wr (wild-dream-576)
            tx = optax.adam(config.learning_rate)
        else:
            # See runs:
            # - https://wandb.ai/skainswo/git-re-basin/runs/k4luj7er (faithful-spaceship-579)
            # - https://wandb.ai/skainswo/git-re-basin/runs/3ru7xy8c (sage-forest-580)
            tx = optax.adamw(config.learning_rate, weight_decay=1e-4)

        train_state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(rngmix(rng, "init"), jnp.zeros((1, 28, 28, 1)))["params"],
            tx=tx,
        )

        for epoch in tqdm(range(config.num_epochs)):
            infos = []
            with timeblock(f"Epoch"):
                batch_ix = random.permutation(rngmix(rng, f"epoch-{epoch}"), num_train_examples).reshape(
                    (-1, config.batch_size))
                for i in range(batch_ix.shape[0]):
                    p = batch_ix[i, :]
                    images_u8 = train_ds["images_u8"][p, :, :, :]
                    labels = train_ds["labels"][p]
                    train_state, info = stuff["step"](train_state, images_u8, labels)
                    infos.append(info)

            train_loss = sum(config.batch_size * x["batch_loss"] for x in infos) / num_train_examples
            train_accuracy = sum(x["num_correct"] for x in infos) / num_train_examples

            # Evaluate train/test loss/accuracy
            with timeblock("Test set eval"):
                test_loss, test_accuracy = stuff["dataset_loss_and_accuracy"](train_state.params, test_ds,
                                                                              10_000)

            params_l2 = tree_map(lambda x: jnp.sqrt(jnp.sum(x ** 2)),
                                 flatten_params({"params_l2": train_state.params}))

            # See https://github.com/wandb/client/issues/3690.
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                # 9
                **params_l2
            })

            # With layer width 512, the MLP is 3.7MB per checkpoint.
            
            with timeblock("model serialization"):
                with artifact.new_file(f"checkpoint{epoch}", mode="wb") as f:
                    f.write(flax.serialization.to_bytes(train_state.params))
        
        with timeblock("model serialization"):
            with artifact.new_file(f"checkpoint{epoch}", mode="wb") as f:
                f.write(flax.serialization.to_bytes(train_state.params))
        # This will be a no-op when config.test is enabled anyhow, since wandb will
        # be initialized with mode="disabled".
        wandb_run.log_artifact(artifact,aliases=[f'seed{args.seed}-{config.noise_type}'])


if __name__ == "__main__":
    main()
