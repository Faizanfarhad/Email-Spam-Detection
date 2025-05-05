import pygame 
import pyperclip 
import sys
import os
import pickle
from gtts import gTTS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model

with open("my_model.pkl", "rb") as f:
    w, b = pickle.load(f)

with open("my_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)



pygame.init()
screen = pygame.display.set_mode((1100,650))
# background  = pygame.image.load('/home/faizan/Documents/email spam-detaction/gui  interface/mike-yukhtenko-wfh8dDlNFOk-unsplash.jpg')
Click = pygame.time.Clock()
font = pygame.font.Font(None,30)
user_text = ''
color = (255,0,0)
active = False
running = True

buttonX = 500 
buttonY = 300
button_width = 153
button_height = 40

latest_prediction_surface = None
latest_prediction_area = None

while running:
    # screen.blit(background,(0,0))
    screen.fill("black")
    text_surface = font.render(user_text,True,(255,0,0))
    input_rect = pygame.Rect(250, 50, 600, 100)
    text_surface1 = font.render('Submit',True,(240, 230, 140))
    text_surface_area = text_surface1.get_rect(topleft=(buttonX + 10, buttonY + 5))
    pygame.draw.rect(screen, "black", input_rect, border_radius=15)
    
    for event in pygame. event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            
            if text_surface_area.collidepoint(event.pos):
                
                print('Submit Text : ' + user_text)
                cleaned_text = model.preprocessed_text(user_text)
                vectorized_input = model.tfidf_vectorizer.transform([cleaned_text]).toarray()
                prediction = model.logistic_regression(vectorized_input, w,b)
                pred = ['Spam ' if prediction > 0.5  else 'Not Spam']
                print('Prediction : ' +  pred[0])
                result_text_surface = font.render(f'Prediction : {pred[0]}',True,(255, 0, 0))
                result_text_surface_area = result_text_surface.get_rect(topleft=(500, 400))
                latest_prediction_surface = font.render(f'Prediction : {pred[0]}', True, (255, 0, 0))
                
                if latest_prediction_surface:
                    tts = gTTS(f'output is {pred[0]}', lang='en')
                    tts.save('output.mp3')
                    sound = pygame.mixer.Sound('output.mp3')
                    sound.play()
                latest_prediction_area = latest_prediction_surface.get_rect(topleft=(500, 400))
                
                
                    
                    
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                # Ctrl+V pressed
                paste_text = pyperclip.paste()
                user_text += paste_text
            
            if event.key == pygame.K_l and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                user_text = ''

            if event.key ==  pygame.K_BACKSPACE: 
                user_text = user_text[:-1] 
            else:
                user_text += event.unicode
                    
                
    
    
    if latest_prediction_surface:
        screen.blit(latest_prediction_surface, latest_prediction_area) 
    
    inform_text_surface = font.render(f'Message Area',True,(255,0,0))
    inform_text_surface_area = inform_text_surface.get_rect(topleft=(400 ,10))
    screen.blit(inform_text_surface,inform_text_surface_area)
    
    
    button_rect = pygame.Rect(buttonX, buttonY, button_width, button_height)
    pygame.draw.rect(screen,(0, 0, 0),button_rect)
    screen.blit(text_surface1, (buttonX + 10, buttonY + 10))
    input_rect.w = max(300,text_surface.get_width() + 30)

    

    
    
    pygame.draw.rect(screen, color, input_rect, 2, border_radius=15)

    screen.blit(text_surface,(input_rect.x + 20 , input_rect.y + 10))
    pygame.display.update()
    Click.tick(20)
pygame.quit()
    
    
