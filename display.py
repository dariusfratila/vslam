import os
import cv2
import pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class Display:
  def __init__(self, width, height):
    pygame.init()
    self.width = width
    self.height = height
    self.window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Alpha SLAM")

  def update_display(self, image):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        return

    surface = pygame.surfarray.pixels3d(self.window)
    surface[:, :, 0:3] = image.swapaxes(0, 1)

    pygame.display.update()
