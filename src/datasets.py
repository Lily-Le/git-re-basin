import numpy as np
import tensorflow_datasets as tfds

def load_cifar10():
  """Return the training and test datasets, as jnp.array's."""
  train_ds_images_u8, train_ds_labels = tfds.as_numpy(
      tfds.load("cifar10", split="train", batch_size=-1, as_supervised=True))
  test_ds_images_u8, test_ds_labels = tfds.as_numpy(
      tfds.load("cifar10", split="test", batch_size=-1, as_supervised=True))
  train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
  test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
  return train_ds, test_ds

def load_cifar10_corrupted(noise_type):
  """Return the training and test datasets, as jnp.array's."""
  train_ds_images_u8, train_ds_labels = tfds.as_numpy(
      tfds.load(f"cifar10_corrupted/{noise_type}", split="train", batch_size=-1, as_supervised=True))
  test_ds_images_u8, test_ds_labels = tfds.as_numpy(
      tfds.load(f"cifar10_corrupted/{noise_type}", split="test", batch_size=-1, as_supervised=True))
  train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
  test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
  return train_ds, test_ds

def load_cifar10_merged(noise_type):
    train1,test1=load_cifar10()
    train2,test2=load_cifar10_corrupted(noise_type)
    
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


def load_cifar100():
  train_ds_images_u8, train_ds_labels = tfds.as_numpy(
      tfds.load("cifar100", split="train", batch_size=-1, as_supervised=True))
  test_ds_images_u8, test_ds_labels = tfds.as_numpy(
      tfds.load("cifar100", split="test", batch_size=-1, as_supervised=True))
  train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
  test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
  return train_ds, test_ds

def _split_cifar(train_ds, label_split: int):
  """Split a CIFAR-ish dataset into two biased subsets."""
  assert train_ds["images_u8"].shape[0] == 50_000
  assert train_ds["labels"].shape[0] == 50_000

  # We randomly permute the training data, just in case there's some kind of
  # non-iid ordering coming out of tfds.
  perm = np.random.default_rng(123).permutation(50_000)
  train_images_u8 = train_ds["images_u8"][perm, :, :, :]
  train_labels = train_ds["labels"][perm]

  # This just so happens to be a clean 25000/25000 split.
  lt_images_u8 = train_images_u8[train_labels < label_split]
  lt_labels = train_labels[train_labels < label_split]
  gte_images_u8 = train_images_u8[train_labels >= label_split]
  gte_labels = train_labels[train_labels >= label_split]
  s1 = {
      "images_u8": np.concatenate((lt_images_u8[:5000], gte_images_u8[5000:]), axis=0),
      "labels": np.concatenate((lt_labels[:5000], gte_labels[5000:]), axis=0)
  }
  s2 = {
      "images_u8": np.concatenate((gte_images_u8[:5000], lt_images_u8[5000:]), axis=0),
      "labels": np.concatenate((gte_labels[:5000], lt_labels[5000:]), axis=0)
  }
  return s1, s2

def load_cifar10_split():
  train_ds, test_ds = load_cifar10()
  s1, s2 = _split_cifar(train_ds, label_split=5)
  return s1, s2, test_ds

def load_cifar100_split():
  train_ds, test_ds = load_cifar100()
  s1, s2 = _split_cifar(train_ds, label_split=50)
  return s1, s2, test_ds
