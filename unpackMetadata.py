def getMetadataDict(*args, **kwargs):
    if 'bHandle' not in kwargs.keys():
        _, bHandle = self.loadNixHandles(*args, **kwargs)
    else:
        bHandle = kwargs['bHandle']

    def unpackMetadata(sec):
        metadata = dict()
        metadata = {prop.name: sec[prop.name] for prop in sec.props}
        if hasattr(sec, 'sections') and len(sec.sections) > 0:
            metadata.update({subsec.name: unpackMetadata(subsec) for subsec in sec.sections})
        return metadata

    return unpackMetadata(bHandle.metadata)
