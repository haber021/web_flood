from django import forms
from .models import Sensor, Municipality, Barangay


class SensorForm(forms.ModelForm):
    class Meta:
        model = Sensor
        fields = [
            'latitude', 'longitude', 'active',
            'municipality', 'barangay', 'description'
        ]
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3, 'class': 'form-control', 'placeholder': 'Optional description or notes about this sensor'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes and placeholders
        self.fields['latitude'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Latitude (e.g., 16.720231)'})
        self.fields['longitude'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Longitude (e.g., 120.351557)'})
        self.fields['active'].widget.attrs.update({'class': 'form-check-input'})
        self.fields['municipality'].widget.attrs.update({'class': 'form-select'})
        self.fields['barangay'].widget.attrs.update({'class': 'form-select'})
        self.fields['barangay'].required = True  # Make barangay required
        self.fields['barangay'].widget.attrs.update({'required': True})

    def clean(self):
        cleaned = super().clean()
        barangay = cleaned.get('barangay')
        lat = cleaned.get('latitude')
        lon = cleaned.get('longitude')
        muni = cleaned.get('municipality')

        # If barangay selected and lat/lon missing, auto-fill from barangay center
        if barangay and (lat is None or lon is None):
            b_lat = getattr(barangay, 'latitude', None)
            b_lon = getattr(barangay, 'longitude', None)
            if b_lat is not None and b_lon is not None:
                cleaned['latitude'] = b_lat
                cleaned['longitude'] = b_lon
                lat, lon = b_lat, b_lon
            else:
                # If barangay has no coordinates, set defaults to avoid errors
                if lat is None:
                    cleaned['latitude'] = 0.0
                if lon is None:
                    cleaned['longitude'] = 0.0

        # Align municipality with selected barangay if not matching
        if barangay and barangay.municipality and (not muni or barangay.municipality_id != getattr(muni, 'id', None)):
            cleaned['municipality'] = barangay.municipality

        # Fallback: if no barangay provided or barangay has no coordinates, auto-fill from municipality coordinates
        if muni and (cleaned.get('latitude') is None or cleaned.get('longitude') is None):
            m_lat = getattr(muni, 'latitude', None)
            m_lon = getattr(muni, 'longitude', None)
            if m_lat is not None and m_lon is not None:
                cleaned['latitude'] = m_lat
                cleaned['longitude'] = m_lon
            else:
                # If municipality has no coordinates, set defaults to avoid errors
                if cleaned.get('latitude') is None:
                    cleaned['latitude'] = 0.0
                if cleaned.get('longitude') is None:
                    cleaned['longitude'] = 0.0

        # Secondary fallback: derive from municipality's barangays if municipality has no coords
        if (cleaned.get('latitude') is None or cleaned.get('longitude') is None) and muni:
            try:
                # Prefer a barangay with defined coordinates
                b = Barangay.objects.filter(municipality=muni, latitude__isnull=False, longitude__isnull=False).first()
                if b:
                    cleaned['latitude'] = b.latitude
                    cleaned['longitude'] = b.longitude
                else:
                    # If no barangays have coordinates, set defaults
                    if cleaned.get('latitude') is None:
                        cleaned['latitude'] = 0.0
                    if cleaned.get('longitude') is None:
                        cleaned['longitude'] = 0.0
            except Exception:
                # Set defaults on any error
                if cleaned.get('latitude') is None:
                    cleaned['latitude'] = 0.0
                if cleaned.get('longitude') is None:
                    cleaned['longitude'] = 0.0

        # Final fallback: set to 0.0 to avoid blocking save (already handled above)
        pass

        return cleaned

    def clean_latitude(self):
        lat = self.cleaned_data.get('latitude')
        # Allow None or empty here; we'll auto-fill from barangay in clean()
        if lat is None or lat == '':
            return None
        try:
            lat = float(lat)
        except ValueError:
            raise forms.ValidationError('Latitude must be a number')
        if lat < -90 or lat > 90:
            raise forms.ValidationError('Latitude must be between -90 and 90')
        return lat

    def clean_longitude(self):
        lon = self.cleaned_data.get('longitude')
        # Allow None or empty here; we'll auto-fill from barangay in clean()
        if lon is None or lon == '':
            return None
        try:
            lon = float(lon)
        except ValueError:
            raise forms.ValidationError('Longitude must be a number')
        if lon < -180 or lon > 180:
            raise forms.ValidationError('Longitude must be between -180 and 180')
        return lon
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User, Group
from .models import FloodAlert, ThresholdSetting, Barangay, Municipality, UserProfile

class FloodAlertForm(forms.ModelForm):
    """Form for creating and editing flood alerts"""

    # Accept HTML datetime-local values (with 'T') and make optional
    predicted_flood_time = forms.DateTimeField(
        required=False,
        input_formats=['%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M']
    )
    scheduled_send_time = forms.DateTimeField(
        required=False,
        input_formats=['%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M']
    )

    # Admin-like date/time selection: separate date and time inputs for clarity
    predicted_month = forms.ChoiceField(
        required=False,
        choices=[],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    predicted_day = forms.ChoiceField(
        required=False,
        choices=[],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    predicted_time = forms.TimeField(
        required=False,
        widget=forms.TimeInput(format='%H:%M', attrs={'class': 'form-control', 'type': 'time', 'step': '60'})
    )
    schedule_month = forms.ChoiceField(
        required=False,
        choices=[],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    schedule_day = forms.ChoiceField(
        required=False,
        choices=[],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    schedule_time = forms.TimeField(
        required=False,
        widget=forms.TimeInput(format='%H:%M', attrs={'class': 'form-control', 'type': 'time', 'step': '60'})
    )

    # Textarea that accepts one action per line; converted to list in clean_actions
    actions = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 5, 'placeholder': 'One recommended action per line'})
    )

    class Meta:
        model = FloodAlert
        fields = ['title', 'description', 'severity_level', 'active', 'predicted_flood_time', 'scheduled_send_time', 'affected_barangays', 'actions']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'severity_level': forms.Select(attrs={'class': 'form-select'}),
            'active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            # Ensure the widget renders and parses the same format as input_formats
            'predicted_flood_time': forms.DateTimeInput(format='%Y-%m-%dT%H:%M', attrs={'class': 'form-control', 'type': 'datetime-local'}),
            'scheduled_send_time': forms.DateTimeInput(format='%Y-%m-%dT%H:%M', attrs={'class': 'form-control', 'type': 'datetime-local'}),
            'affected_barangays': forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Allow creating alerts without manually selecting barangays; server will compute affected areas.
        if 'affected_barangays' in self.fields:
            self.fields['affected_barangays'].required = False
        # Optional helper text mirroring admin guidance
        if 'actions' in self.fields:
            self.fields['actions'].help_text = 'Enter one recommended action per line.'
        # Pre-populate actions textarea from instance list
        try:
            instance = kwargs.get('instance') or getattr(self, 'instance', None)
            if instance and getattr(instance, 'actions_list', None):
                self.initial['actions'] = '\n'.join(instance.actions_list)
        except Exception:
            pass
        # Hide raw DateTime fields in favor of split date/time controls
        if 'predicted_flood_time' in self.fields:
            self.fields['predicted_flood_time'].widget = forms.HiddenInput()
        if 'scheduled_send_time' in self.fields:
            self.fields['scheduled_send_time'].widget = forms.HiddenInput()
        # Initialize month/day choices and split date/time fields from instance values
        try:
            from calendar import month_name
            month_choices = [('', 'Month')] + [(str(i), month_name[i]) for i in range(1, 13)]
            day_choices = [('', 'Day')] + [(str(i), str(i)) for i in range(1, 32)]
            self.fields['predicted_month'].choices = month_choices
            self.fields['predicted_day'].choices = day_choices
            self.fields['schedule_month'].choices = month_choices
            self.fields['schedule_day'].choices = day_choices
        except Exception:
            pass
        try:
            instance = getattr(self, 'instance', None)
            from django.utils import timezone as dj_tz
            if instance and getattr(instance, 'predicted_flood_time', None):
                ts = instance.predicted_flood_time
                if dj_tz.is_aware(ts):
                    ts = dj_tz.localtime(ts)
                self.initial.setdefault('predicted_month', str(ts.month))
                self.initial.setdefault('predicted_day', str(ts.day))
                self.initial.setdefault('predicted_time', ts.time().replace(second=0, microsecond=0))
            if instance and getattr(instance, 'scheduled_send_time', None):
                ts2 = instance.scheduled_send_time
                if dj_tz.is_aware(ts2):
                    ts2 = dj_tz.localtime(ts2)
                self.initial.setdefault('schedule_month', str(ts2.month))
                self.initial.setdefault('schedule_day', str(ts2.day))
                self.initial.setdefault('schedule_time', ts2.time().replace(second=0, microsecond=0))
        except Exception:
            pass

    def clean_actions(self):
        """Convert textarea content to a list of actions for JSONField compatibility."""
        actions = self.cleaned_data.get('actions')
        if isinstance(actions, str):
            lines = [line.strip() for line in actions.splitlines() if line.strip()]
            return lines
        return actions or []

    def clean(self):
        cleaned = super().clean()
        from datetime import datetime
        from django.utils import timezone as dj_tz

        # Combine predicted month/day/time into model field (default year = current year)
        p_month = cleaned.get('predicted_month')
        p_day = cleaned.get('predicted_day')
        p_time = cleaned.get('predicted_time')
        if p_month and p_day and p_time:
            try:
                year = dj_tz.localdate().year
                dt = datetime(year=int(year), month=int(p_month), day=int(p_day),
                              hour=p_time.hour, minute=p_time.minute)
                if dj_tz.is_naive(dt):
                    dt = dj_tz.make_aware(dt, dj_tz.get_default_timezone())
                # Reject past datetimes
                now = dj_tz.now()
                if dt < now:
                    self.add_error('predicted_day', 'Predicted time cannot be in the past.')
                    self.add_error('predicted_time', 'Choose a future time.')
                else:
                    cleaned['predicted_flood_time'] = dt
            except Exception:
                # Leave field unset on invalid date (e.g., Feb 30)
                pass

        # Combine schedule month/day/time (default year = current year)
        s_month = cleaned.get('schedule_month')
        s_day = cleaned.get('schedule_day')
        s_time = cleaned.get('schedule_time')
        if s_month and s_day and s_time:
            try:
                year2 = dj_tz.localdate().year
                dt2 = datetime(year=int(year2), month=int(s_month), day=int(s_day),
                               hour=s_time.hour, minute=s_time.minute)
                if dj_tz.is_naive(dt2):
                    dt2 = dj_tz.make_aware(dt2, dj_tz.get_default_timezone())
                # Reject past datetimes
                now2 = dj_tz.now()
                if dt2 < now2:
                    self.add_error('schedule_day', 'Schedule time cannot be in the past.')
                    self.add_error('schedule_time', 'Choose a future time.')
                else:
                    cleaned['scheduled_send_time'] = dt2
            except Exception:
                pass
        return cleaned

class ThresholdSettingForm(forms.ModelForm):
    """Form for creating and editing threshold settings"""
    class Meta:
        model = ThresholdSetting
        fields = ['parameter', 'advisory_threshold', 'watch_threshold', 'warning_threshold', 'emergency_threshold', 'catastrophic_threshold', 'unit']
        widgets = {
            'parameter': forms.Select(attrs={'class': 'form-select'}),
            'advisory_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'watch_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'warning_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'emergency_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'catastrophic_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'unit': forms.TextInput(attrs={'class': 'form-control'}),
        }

    def clean(self):
        cleaned = super().clean()
        a = cleaned.get('advisory_threshold')
        w = cleaned.get('watch_threshold')
        wn = cleaned.get('warning_threshold')
        e = cleaned.get('emergency_threshold')
        c = cleaned.get('catastrophic_threshold')
        msg = 'Thresholds must be strictly increasing: Advisory < Watch < Warning < Emergency < Catastrophic.'
        if None not in (a, w, wn, e, c):
            if not (a < w < wn < e < c):
                self.add_error('advisory_threshold', msg)
                self.add_error('watch_threshold', msg)
                self.add_error('warning_threshold', msg)
                self.add_error('emergency_threshold', msg)
                self.add_error('catastrophic_threshold', msg)
        return cleaned

class BarangaySearchForm(forms.Form):
    """Form for searching and filtering barangays"""
    name = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Search by name'})
    )
    
    severity_level = forms.ChoiceField(
        choices=[
            ('', 'All Severity Levels'),
            (1, 'Advisory'),
            (2, 'Watch'),
            (3, 'Warning'),
            (4, 'Emergency'),
            (5, 'Catastrophic'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )

class RegisterForm(UserCreationForm):
    """Registration form for new users"""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter your email'})
    )
    
    first_name = forms.CharField(
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'First name'})
    )
    
    last_name = forms.CharField(
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Last name'})
    )
    
    role = forms.ChoiceField(
        choices=[
            ('viewer', 'Data Viewer - Can only view data'),
            ('officer', 'Municipal Officer - Can view data and create alerts'),
            ('operator', 'System Operator - Can manage sensors and view data'),
            ('manager', 'Flood Manager - Can manage alerts and sensors'),
        ],
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    municipality = forms.ModelChoiceField(
        queryset=Municipality.objects.filter(is_active=True),
        required=False,
        empty_label="Select municipality (optional)",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    phone_number = forms.CharField(
        max_length=20,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Phone number (optional)'})
    )
    
    receive_alerts = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    receive_sms = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    receive_email = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']
        
    def __init__(self, *args, **kwargs):
        super(RegisterForm, self).__init__(*args, **kwargs)
        # Add Bootstrap classes to form fields
        self.fields['username'].widget.attrs['class'] = 'form-control'
        self.fields['username'].widget.attrs['placeholder'] = 'Choose a username'
        self.fields['password1'].widget.attrs['class'] = 'form-control'
        self.fields['password1'].widget.attrs['placeholder'] = 'Create a password'
        self.fields['password2'].widget.attrs['class'] = 'form-control'
        self.fields['password2'].widget.attrs['placeholder'] = 'Confirm your password'
        
    def save(self, commit=True):
        user = super(RegisterForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
            
            # Create or update the user profile
            if hasattr(user, 'profile'):
                profile = user.profile
            else:
                profile = UserProfile(user=user)
                
            # Update profile fields
            profile.role = self.cleaned_data['role']
            profile.municipality = self.cleaned_data.get('municipality')
            profile.phone_number = self.cleaned_data.get('phone_number')
            profile.receive_alerts = self.cleaned_data.get('receive_alerts', True)
            profile.receive_sms = self.cleaned_data.get('receive_sms', False)
            profile.receive_email = self.cleaned_data.get('receive_email', True)
            profile.save()
            
            # Add user to appropriate group
            if self.cleaned_data['role'] == 'manager':
                group = Group.objects.get(name='Flood Managers')
            elif self.cleaned_data['role'] == 'officer':
                group = Group.objects.get(name='Municipal Officers')
            elif self.cleaned_data['role'] == 'operator':
                group = Group.objects.get(name='System Operators')
            else:  # default to viewer
                group = Group.objects.get(name='Viewers')
                
            user.groups.clear()
            user.groups.add(group)
            
        return user


class UserProfileForm(forms.ModelForm):
    """Form for editing user profiles"""
    first_name = forms.CharField(
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    last_name = forms.CharField(
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = UserProfile
        fields = ['role', 'municipality', 'barangay', 'phone_number', 
                 'receive_alerts', 'receive_sms', 'receive_email']
        widgets = {
            'role': forms.Select(attrs={'class': 'form-select'}),
            'municipality': forms.Select(attrs={'class': 'form-select'}),
            'barangay': forms.Select(attrs={'class': 'form-select'}),
            'phone_number': forms.TextInput(attrs={'class': 'form-control'}),
            'receive_alerts': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'receive_sms': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'receive_email': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
        
    def __init__(self, *args, **kwargs):
        # Get the user instance to populate initial values
        user = kwargs.pop('user', None)
        super(UserProfileForm, self).__init__(*args, **kwargs)
        
        if user:
            self.fields['first_name'].initial = user.first_name
            self.fields['last_name'].initial = user.last_name
            self.fields['email'].initial = user.email
            
    def save(self, user=None, commit=True):
        profile = super(UserProfileForm, self).save(commit=False)
        
        if user:
            # Update the User model fields
            user.first_name = self.cleaned_data['first_name']
            user.last_name = self.cleaned_data['last_name']
            user.email = self.cleaned_data['email']
            
            if commit:
                user.save()
                
                # Update user's group based on role
                user.groups.clear()
                
                if profile.role == 'admin':
                    group = Group.objects.get(name='Administrators')
                elif profile.role == 'manager':
                    group = Group.objects.get(name='Flood Managers')
                elif profile.role == 'officer':
                    group = Group.objects.get(name='Municipal Officers')
                elif profile.role == 'operator':
                    group = Group.objects.get(name='System Operators')
                else:  # default to viewer
                    group = Group.objects.get(name='Viewers')
                    
                user.groups.add(group)
        
        if commit:
            profile.save()
            
        return profile


class MunicipalityForm(forms.ModelForm):
    """Form to create/edit Municipality entries"""
    class Meta:
        model = Municipality
        fields = ['name', 'province', 'latitude', 'longitude', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Municipality name'}),
            'province': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Province or region (optional)'}),
            'latitude': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Latitude (optional)'}),
            'longitude': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Longitude (optional)'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'})
        }

    def clean_latitude(self):
        lat = self.cleaned_data.get('latitude')
        if lat in (None, ''):
            return None
        try:
            lat = float(lat)
        except (ValueError, TypeError):
            raise forms.ValidationError('Latitude must be a number')
        if lat < -90 or lat > 90:
            raise forms.ValidationError('Latitude must be between -90 and 90')
        return lat

    def clean_longitude(self):
        lon = self.cleaned_data.get('longitude')
        if lon in (None, ''):
            return None
        try:
            lon = float(lon)
        except (ValueError, TypeError):
            raise forms.ValidationError('Longitude must be a number')
        if lon < -180 or lon > 180:
            raise forms.ValidationError('Longitude must be between -180 and 180')
        return lon


class BarangayForm(forms.ModelForm):
    """Form to create/edit Barangay entries"""
    class Meta:
        model = Barangay
        fields = ['name', 'municipality', 'population', 'latitude', 'longitude', 'contact_person', 'contact_number']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Barangay name'}),
            'municipality': forms.Select(attrs={'class': 'form-select'}),
            'population': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Estimated population (optional)'}),
            'latitude': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Latitude (optional)'}),
            'longitude': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Longitude (optional)'}),
            'contact_person': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Contact person (optional)'}),
            'contact_number': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Contact number (optional)'}),
        }

    def clean_population(self):
        pop = self.cleaned_data.get('population')
        if pop is not None and pop < 0:
            raise forms.ValidationError('Population cannot be negative')
        return pop

