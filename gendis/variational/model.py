from .models.conv_vae import Conv_VAE


def make_img_model(
    channels=1,
    height=28,
    width=28,
    lr=5e-3,
    lr_scheduler='cosine',
    hidden_size=32,
    alpha=1024,
    batch_size=144,
    save_images=False,
    save_path=None,
):
    model = Conv_VAE(
        channels=channels,
        height=height,
        width=width,
        lr=lr,
        lr_scheduler=lr_scheduler,
        hidden_size=hidden_size,
        alpha=alpha,
        batch_size=batch_size,
        save_images=save_images,
        save_path=save_path,
    )
    return model
