"""Minimal Visdom stub so guided_diffusion imports without a real Visdom server."""

class Visdom:
    def __init__(self, *args, **kwargs):
        self.server = kwargs.get("server")
        self.port = kwargs.get("port")

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop
