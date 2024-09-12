
import jax
import jax.numpy as jp

def norm(v, axis=-1, keepdims=False, eps=0.0):
  return jp.sqrt((v*v).sum(axis, keepdims=keepdims).clip(eps))

def normalize(v, axis=-1, eps=1e-20):
  return v/norm(v, axis, keepdims=True, eps=eps)

import io
import base64
import time
from functools import partial
from typing import NamedTuple
import subprocess

import PIL
import numpy as np
import matplotlib.pylab as pl

# from IPython.display import display, Image, HTML

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1) * 255)
  return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()


def imshow(a=[], fmt='jpeg', display=[]):
  return display(Image(data=imencode(a, fmt)))


class VideoWriter:
  def __init__(self, filename='_autoplay.mp4', fps=30.0):
    self.ffmpeg = None
    self.filename = filename
    self.fps = fps
    self.view = display(display_id=True)
    self.last_preview_time = 0.0

  def add(self, img):
    img = np.asarray(img)
    h, w = img.shape[:2]
    if self.ffmpeg is None:
      self.ffmpeg = self._open(w, h)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1) * 255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.ffmpeg.stdin.write(img.tobytes())
    t = time.time()
    if self.view and t - self.last_preview_time > 1:
      self.last_preview_time = t
      imshow(img, display=self.view.update)

  def __call__(self, img):
    return self.add(img)

  def _open(self, w, h):
    cmd = f'''ffmpeg -y -f rawvideo -vcodec rawvideo -s {w}x{h}
      -pix_fmt rgb24 -r {self.fps} -i - -pix_fmt yuv420p 
      -c:v libx264 -crf 20 {self.filename}'''.split()
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

  def close(self):
    if self.ffmpeg:
      self.ffmpeg.stdin.close()
      self.ffmpeg.wait()
      self.ffmpeg = None

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
    if self.filename == '_autoplay.mp4':
      self.show()

  def show(self):
    self.close()
    if not self.view:
      return
    b64 = base64.b64encode(open(self.filename, 'rb').read()).decode('utf8')
    s = f'''<video controls loop>
 <source src="data:video/mp4;base64,{b64}" type="video/mp4">
 Your browser does not support the video tag.</video>'''
    self.view.update(HTML(s))


def animate(f, duration_sec, fps=60):
  with VideoWriter(fps=fps) as vid:
    for t in jp.linspace(0, 1, int(duration_sec * fps)):
      vid(f(t))


class Balls(NamedTuple):
  pos: jp.ndarray
  color: jp.ndarray

def balls_sdf(balls, p, ball_r=0.1):
  dists = norm(p-balls.pos)-ball_r
  return dists.min()

def scene_sdf(balls, p, ball_r=0.1, c=8.0):
  dists = norm(p-balls.pos)-ball_r
  balls_dist = -jax.nn.logsumexp(-dists*c)/c  # softmin
  floor_dist = p[1]+3.0  # floor is at y==-3.0
  return jp.minimum(balls_dist, floor_dist)

def raycast(sdf, p0, dir, step_n=50):
  def f(_, p):
    return p+sdf(p)*dir
  return jax.lax.fori_loop(0, step_n, f, p0)

def camera_rays(forward, view_size, fx=0.2):
  world_up = jp.array([0., 2., 0.])
  right = jp.cross(forward, world_up)
  down = jp.cross(right, forward)
  R = normalize(jp.vstack([right, down, forward]))
  w, h = view_size
  fy = fx/w*h
  y, x = jp.mgrid[fy:-fy:h*1j, -fx:fx:w*1j].reshape(2, -1)
  return normalize(jp.c_[x, y, jp.ones_like(x)]) @ R

def cast_shadow(sdf, light_dir, p0, step_n=50, hardness=8.0):
  def f(_, carry):
    t, shadow = carry
    h = sdf(p0+light_dir*t)
    return t+h, jp.clip(hardness*h/t, 0.0, shadow)
  return jax.lax.fori_loop(0, step_n, f, (1e-2, 1.0))[1]



def shade_f(surf_color, shadow, raw_normal, ray_dir, light_dir):
  ambient = norm(raw_normal)
  normal = raw_normal/ambient
  diffuse = normal.dot(light_dir).clip(0.0)*shadow
  half = normalize(light_dir-ray_dir)
  spec = 0.3 * shadow * half.dot(normal).clip(0.0)**200.0
  light = 0.7*diffuse+0.2*ambient
  return surf_color*light + spec



def scene_sdf(balls, p, ball_r=0.1, c=8.0, with_color=False):
  dists = norm(p-balls.pos)-ball_r
  balls_dist = -jax.nn.logsumexp(-dists*c)/c
  floor_dist = p[1]+3.0  # floor is at y==-3.0
  min_dist = jp.minimum(balls_dist, floor_dist)
  if not with_color:
    return min_dist
  x, y, z = jp.tanh(jp.sin(p*jp.pi)*20.0)
  floor_color = jp.ones(3) # (0.5+(x*z)*0.1)*jp.ones(3)
  balls_color = jax.nn.softmax(-dists*c) @ balls.color
  color = jp.choose(jp.int32(floor_dist < balls_dist),
            [balls_color, floor_color], mode='clip')
  return min_dist, color




if __name__ == '__main__':

  matplotlib.use("Qt5Agg")

  p = (np.random.random((1000, 3)))*10.0-5
  p[:, 1] += 2.5
  pos = jp.array(p)
  c = np.random.random((1000, 3))
  color = jp.array(c)
  balls = Balls(pos, color)

  # show_slice(partial(balls_sdf, balls), z=0.0)

  w, h = 640, 640
  pos0 = jp.float32([5,30,35])
  pos0 = jp.float32([2.5, 15, 17.5])
  ray_dir = camera_rays(-pos0, view_size=(w, h))
  sdf = partial(scene_sdf, balls)
  hit_pos = jax.vmap(partial(raycast, sdf, pos0))(ray_dir)
  # pl.imshow(hit_pos.reshape(h, w, 3)%1.0)
  raw_normal = jax.vmap(jax.grad(sdf))(hit_pos)
  # pl.imshow(raw_normal.reshape(h, w, 3))
  light_dir = normalize(jp.array([1.1, 1.0, 0.2]))
  shadow = jax.vmap(partial(cast_shadow, sdf, light_dir))(hit_pos)
  # pl.imshow(shadow.reshape(h, w))
  f = partial(shade_f, jp.ones(3), light_dir=light_dir)
  frame = jax.vmap(f)(shadow, raw_normal, ray_dir)
  frame = frame ** (1.0 / 2.2)  # gamma correction
  # pl.imshow(frame.reshape(h, w, 3))
  color_sdf = partial(scene_sdf, balls, with_color=True)
  _, surf_color = jax.vmap(color_sdf)(hit_pos)
  f = partial(shade_f, light_dir=light_dir)
  frame = jax.vmap(f)(surf_color, shadow, raw_normal, ray_dir)
  frame = frame**(1.0/2.2)  # gamma correction
  pl.figure(figsize=(8, 8))
  pl.imshow(frame.reshape(h, w, 3))













