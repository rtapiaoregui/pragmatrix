#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:25:38 2018

@author: rita
"""


from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, EqualTo


class InputForm(FlaskForm):
    new_text = StringField('Text to test',
                           validators=[DataRequired(), Length(min=3000, max=6000)])

    submit = SubmitField('Submit text')
